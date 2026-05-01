import os
import pandas as pd

from components.model_scripts.make_server_farms import create_server_farms
from components.model_scripts.make_user_workloads import generate_graph
from components.models.vm import Vm
from components.models.server import Server
from components.models.server_farm import Server_Farm
from components.models.task import Task
from components.models.job import Job
from components.timeline import TimelineEvent
from env.cloud_scheduling_v0 import CloudSchedulingEnv
from schedulers.marl.maddpg.MADDPG import MADDPG

# Monkey-patch Vm to fix capacity vs used
def vm_init(self, id: int, cpu: float, ram: float):
    self.id = id
    self.capacity_cpu = cpu
    self.capacity_ram = ram
    self.used_cpu = 0.0
    self.used_ram = 0.0
    self.status = 0  # 0: off, 1: occupied
    self.hosted_task = None

def vm_host_task(self, task):
    if self.used_cpu + task.cpu > self.capacity_cpu or self.used_ram + task.ram > self.capacity_ram:
        return False
    self.hosted_task = task
    self.used_cpu += task.cpu
    self.used_ram += task.ram
    self.status = 1  # Set VM status to 1: occupied
    return True

def vm_release_task(self):
    assert self.hosted_task is not None, "VM has not hosted task, expect a hosted task"
    if self.status == 1 and self.hosted_task:
        self.used_cpu -= self.hosted_task.cpu
        self.used_ram -= self.hosted_task.ram
        self.status = 0
        hosted_task_id = self.hosted_task.id
        vm_id = self.id
        self.hosted_task = None
    return hosted_task_id, vm_id

Vm.__init__ = vm_init
Vm.host_task = vm_host_task
Vm.release_task = vm_release_task

# Monkey-patch Server to increase precision and change static power condition
def cpu_utilization_rate(self):
    total_vms_cpu = sum(vm.used_cpu for vm in self.vms.values() if vm.status == 1)
    cpu_utilization_rate = total_vms_cpu / self.c_cpu
    return round(cpu_utilization_rate, 4)  # Increased precision

def static_power(self):
    return 0.035 if self.cpu_utilization_rate >= 0 else 0  # Changed to >= 0

def host_task_in_server(self, task):
    available_vm_id = next((vm.id for vm in self.vms.values() if vm.status == 0), None)
    if available_vm_id: # take the available vm id
        if self.vms[available_vm_id].host_task(task): # check and host
            task.vm_id = available_vm_id
            task.server_id = self.id
            task.server_farm_id = self.server_farm_id
            self.current_cpu_usage += task.cpu
            self.current_ram_usage += task.ram
            return True, task # True for successful task hosting
        return False # False for rejected task

Server.cpu_utilization_rate = property(cpu_utilization_rate)
Server.static_power = property(static_power)
Server.host_task_in_server = host_task_in_server


def build_servers_from_farm_graph(farm_graph, farm_id: int):
    servers = []
    for vertex in farm_graph.vs:
        server_id = int(vertex["name"])
        vm_cpu, vm_ram = next(iter(vertex["VM_CPU_and_MEM"]))
        c_cpu, c_ram = next(iter(vertex["Cumulative_Server_CPU_and_MEM"]))
        alpha, beta = next(iter(vertex["Power_Consumption_Coefficients"]))

        # VM IDs must be truthy in your Server.host_task_in_server logic,
        # so use 1..v rather than 0..v-1.
        vm_ids = vertex["VM_type"]
        vms = [Vm(id=vm_id, cpu=vm_cpu, ram=vm_ram) for vm_id in vm_ids]

        server = Server(
            id=server_id,
            server_farm_id=farm_id,
            vms=vms,
            c_cpu=c_cpu,
            c_ram=c_ram,
            alpha=alpha,
            beta=beta,
        )
        servers.append(server)

    return servers


def select_fittable_job_and_task(csv_path, cpu_limit=0.95, ram_limit=0.95):
    df = pd.read_csv(
        csv_path,
        usecols=["Job ID", "Task Index", "Resource Request CPU", "Resource Request RAM"],
    )
    df["Resource Request CPU"] = df["Resource Request CPU"].astype(float)
    df["Resource Request RAM"] = df["Resource Request RAM"].astype(float)

    for job_id, group in df.groupby("Job ID"):
        eligible = group[
            (group["Resource Request CPU"] <= cpu_limit)
            & (group["Resource Request RAM"] <= ram_limit)
        ]
        if not eligible.empty:
            row = eligible.iloc[0]
            return int(job_id), int(row["Task Index"]), float(row["Resource Request CPU"]), float(row["Resource Request RAM"])

    raise RuntimeError(
        "No task in the dataset fits the server/vm allocation constraint."
    )


def build_job_from_resources(job_id, task_resources):
    dag = generate_graph((len(task_resources), task_resources))

    tasks = []
    for vertex in dag.vs:
        cpu, ram = next(iter(vertex["Required_CPU_and_MEM"]))
        task = Task(
            id=str(vertex["name"]),
            job_id=job_id,
            cpu=cpu,
            ram=ram,
            status=3,
            runtime=1.0,
        )
        tasks.append(task)

    job = Job(
        id=job_id,
        dag=dag,
        tasks=tasks,
        num_tasks=len(tasks),
        time_arrived=0.0,
    )
    return job


def main(job_id_input=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "helper", "jobs_dataset", "google_cluster_trace.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if job_id_input is None:
        job_id, task_index, task_cpu, task_ram = select_fittable_job_and_task(dataset_path)
        print(f"Selected job_id={job_id}, task_index={task_index}, task cpu={task_cpu}, ram={task_ram}")
    else:
        # Find a task from the specified job_id that fits
        df = pd.read_csv(
            dataset_path,
            usecols=["Job ID", "Task Index", "Resource Request CPU", "Resource Request RAM"],
        )
        df["Resource Request CPU"] = df["Resource Request CPU"].astype(float)
        df["Resource Request RAM"] = df["Resource Request RAM"].astype(float)
        job_group = df[df["Job ID"] == job_id_input]
        if job_group.empty:
            raise ValueError(f"Job ID {job_id_input} not found in dataset.")
        eligible = job_group[
            (job_group["Resource Request CPU"] <= 0.95) & (job_group["Resource Request RAM"] <= 0.95)
        ]
        if eligible.empty:
            raise ValueError(f"No task in job {job_id_input} fits the constraints.")
        row = eligible.iloc[0]
        job_id = int(row["Job ID"])
        task_index = int(row["Task Index"])
        task_cpu = float(row["Resource Request CPU"])
        task_ram = float(row["Resource Request RAM"])
        print(f"Using specified job_id={job_id}, task_index={task_index}, task cpu={task_cpu}, ram={task_ram}")

    # Preserve original task size; the RL model works with task values in the dataset range.
    print(f"Task size preserved: cpu={task_cpu}, ram={task_ram}")

    NUM_FARMS = 30
    SERVERS_PER_FARM = 8
    total_servers = NUM_FARMS * SERVERS_PER_FARM

    farm_graphs = create_server_farms(total_servers=total_servers, num_farms=NUM_FARMS)
    server_farms = [
        Server_Farm(
            id=farm_id,
            graph=farm_graph,
            servers=build_servers_from_farm_graph(farm_graph, farm_id),
            num_servers=len(farm_graph.vs),
        )
        for farm_id, farm_graph in enumerate(farm_graphs)
    ]

    print(f"Architecture: {NUM_FARMS} server farms, {SERVERS_PER_FARM} servers per farm, 20 VMs per server")

    # Build a job from the selected real task values
    task_resources = [(task_cpu, task_ram)]
    job = build_job_from_resources(job_id=job_id, task_resources=task_resources)
    ready_tasks = job.get_ready_tasks()

    if not ready_tasks:
        raise RuntimeError("No ready task found in the selected job.")

    task = ready_tasks[0]
    print(
        f"Task details:\n"
        f"  task.id={task.id}\n"
        f"  task.job_id={task.job_id}\n"
        f"  task.cpu={task.cpu}\n"
        f"  task.ram={task.ram}\n"
        f"  task.status={task.status}"
    )

    env = CloudSchedulingEnv(num_jobs=1, num_server_farms=NUM_FARMS, num_servers=total_servers)
    env.reset(seed=42)

    # Make the environment use our selected task instead of the random job generated by reset.
    env.wall_time = 0.0
    env.timeline.reset()
    env.jobs = {job.id: job}
    env.active_job_ids = [job.id]
    env.completed_job_ids = set()
    env.rejected_job_ids = set()
    env.rejected_tasks_count = 0
    env.task_rejected_status = False
    env.scheduled_tasks = set()
    env.scheduled_task_cpu = task.cpu
    env.scheduled_task_ram = task.ram
    env.scheduled_task_deadline = task.runtime
    env.prev_server_farm_reward = 0
    env.prev_server_reward = 0
    env.schedulable_tasks = True
    env.server_farm_id = 0
    env.server_id = 0

    env.timeline.push(
        0.0,
        TimelineEvent(
            TimelineEvent.Type.TASK_ARRIVAL,
            data={"task_arrival": task},
        ),
    )

    # Keep server IDs as strings because env._take_action uses string lookup for servers.
    for farm in env.server_farms.values():
        for server in farm.servers.values():
            server.id = str(server.id)

    dim_info = {}
    for agent_id in env.agents:
        obs_space = env.observation_space(agent_id)
        obs_shape = {key: space.shape for key, space in obs_space.spaces.items()}
        action_dim = env.action_space(agent_id).n
        dim_info[agent_id] = {
            'obs_shape': obs_shape,
            'action_dim': action_dim,
        }

    model_path = os.path.join(project_root, 'results', 'maddpg', 'model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    maddpg = MADDPG.load(
        dim_info,
        model_path,
        capacity=int(1e6),
        batch_size=1024,
        actor_lr=0.0005,
        critic_lr=0.0005,
    )

    obs = {agent: env._get_observation(agent) for agent in env.agents}
    actions = maddpg.select_action(obs)
    print(f"RL selected actions: {actions}")

    next_obs, reward, terminated, truncated, infos = env.step(actions)
    if task.server_id is None:
        print("RL policy could not schedule the task.")
        print("Env info:", infos)
        return

    assigned_vm = env.server_farms[task.server_farm_id].servers[task.server_id].vms[task.vm_id]
    assigned_server = env.server_farms[task.server_farm_id].servers[task.server_id]
    assigned_farm = env.server_farms[task.server_farm_id]
    print("Assignment result using trained RL model:")
    print(f"  server_farm_id = {task.server_farm_id}")
    print(f"  server_id = {task.server_id}")
    print(f"  vm_id = {task.vm_id}")
    print(f"  vm.status = {assigned_vm.status}")
    print(f"  vm.hosted_task = {assigned_vm.hosted_task.id}")
    print(f"  reward = {reward}")
    print(f"  price = {infos['server_farm']['price']}")
    print(f"  server CPU utilization after assignment = {assigned_server.cpu_utilization_rate}")
    print(f"  server total power after assignment = {assigned_server.total_power}")
    print(f"  server farm price after assignment = {assigned_farm.get_price}")


if __name__ == "__main__":
    import sys
    job_id_input = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(job_id_input)