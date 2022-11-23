import numpy as np
from tabulate import tabulate
from tqdm import tqdm

np.random.seed(2022)


class Server:
    def __init__(self, means):
        self.cores = [Core(i, False) for i in means]
        self.queue = Queue()


class Core:
    def __init__(self, mean, sch):
        self.mean = mean
        self.busy = False
        self.job_assigned = None
        self.time_remaining = 0
        self.sch = sch

    def assign_job(self, job):
        self.job_assigned = job
        service_time = np.random.exponential(self.mean)
        if self.sch:
            self.job_assigned.sch_time = round(service_time, 4)
        self.time_remaining = round(service_time, 4)
        self.busy = True


class Scheduler:
    def __init__(self, rate):
        self.core = Core(1 / rate, True)
        self.queue = Queue()


class Job:
    def __init__(self, job_type, inter_arrival, time_limit):
        self.type = job_type
        self.inter_arrival = inter_arrival
        self.inter_arrival_remaining = inter_arrival
        self.time_limit = time_limit
        self.time_remaining = time_limit
        self.arrival_time = 0
        self.finish_time = 0
        self.sch_time = 0
        self.processed = False


class Queue:
    def __init__(self):
        self.queue = []
        self.records = []

    def empty(self):
        return len(self.queue) == 0

    def append(self, data):
        self.queue.append(data)

    def length(self):
        return len(self.queue)

    def pop(self):
        element = 0
        for i in range(len(self.queue)):
            if self.queue[i].type == 1:
                element = i
                break
        item = self.queue[element]
        self.queue.remove(item)
        return item

    def record(self, inc):
        self.records.append([self.length(), inc])


def jobs_generate():
    inter_arrival = np.random.exponential(1 / lam, job_size)
    types = np.random.choice([1, 2], job_size, p=[0.1, 0.9])
    time_limits = np.random.exponential(alpha, job_size)
    jobs_list = [Job(types[i], round(inter_arrival[i], 4), round(time_limits[i], 4)) for i in range(job_size)]
    return jobs_list


def simulation_ends(c, scheduler, servers):
    jobs_arrived = c == job_size
    servers_free = True
    for i in servers:
        for j in i.cores:
            servers_free = servers_free and (not j.busy)
    servers_free = servers_free and (not scheduler.core.busy)
    return jobs_arrived and servers_free


def time_inc(c, jobs, scheduler, servers):
    comp = []
    if c < job_size:
        comp.append(jobs[c].inter_arrival_remaining)
    if not scheduler.queue.empty():
        comp.append(min(j.time_remaining for j in scheduler.queue.queue))
    if scheduler.core.busy:
        comp.append(scheduler.core.time_remaining)
        comp.append(scheduler.core.job_assigned.time_remaining)
    for i in servers:
        if not i.queue.empty():
            comp.append(min(j.time_remaining for j in i.queue.queue))
        for j in i.cores:
            if j.busy:
                comp.append(j.time_remaining)
    return min(comp)


def server_choose(servers):
    min_len = float('inf')
    for i in servers:
        if i.queue.length() < min_len:
            min_len = i.queue.length()
    servers_to_select = [i for i in range(len(servers)) if servers[i].queue.length() == min_len]
    selected = np.random.choice(servers_to_select)
    return selected


def sch_queue_update(q, inc, job, arrived):
    sch_rem = []
    for e in q.queue:
        if arrived and job == e:
            continue
        e.time_remaining -= inc
        if e.time_remaining == 0:
            e.finish_time = e.arrival_time + e.time_limit
            sch_rem.append(e)
    for e in sch_rem:
        scheduler.queue.queue.remove(e)


def sch_core_update(scheduler, inc, servers):
    sent = None
    server_to_send = None
    if scheduler.core.busy:
        scheduler.core.time_remaining -= inc
        scheduler.core.job_assigned.time_remaining -= inc
        if scheduler.core.job_assigned.time_remaining == 0 and scheduler.core.time_remaining > 0:
            scheduler.core.job_assigned.finish_time = scheduler.core.job_assigned.arrival_time + scheduler.core.job_assigned.time_limit
            scheduler.core.busy = False
            scheduler.core.job_assigned = None
        elif scheduler.core.time_remaining == 0:
            server_to_send = server_choose(servers)
            servers[server_to_send].queue.append(scheduler.core.job_assigned)
            sent = scheduler.core.job_assigned
            scheduler.core.busy = False
            scheduler.core.job_assigned = None
    if not scheduler.core.busy:
        if not scheduler.queue.empty():
            element = scheduler.queue.pop()
            scheduler.core.assign_job(element)
    return sent, server_to_send


def server_cores_update(server, inc, t):
    for j in server.cores:
        if j.busy:
            j.time_remaining -= inc
            if j.time_remaining == 0:
                j.job_assigned.finish_time = t
                j.job_assigned.processed = True
                j.busy = False
                j.job_assigned = None
        if not j.busy:
            if not server.queue.empty():
                element = server.queue.pop()
                j.assign_job(element)


def server_queue_update(q, inc, sent):
    q_rem = []
    for e in q.queue:
        if sent is not None and e == sent:
            continue
        e.time_remaining -= inc
        if e.time_remaining == 0:
            e.finish_time = e.arrival_time + e.time_limit
            q_rem.append(e)
    for e in q_rem:
        q.queue.remove(e)


def queue_avg(queue, t):
    sum_len = 0
    for r in queue.records:
        sum_len += (r[0] * r[1])
    return sum_len / t


def simulate(jobs, scheduler, servers):
    pbar = tqdm(total=job_size)
    t = 0
    c = 0
    while not simulation_ends(c, scheduler, servers):
        arrived = False
        inc = time_inc(c, jobs, scheduler, servers)
        t += inc
        scheduler.queue.record(inc)
        for i in range(len(servers)):
            servers[i].queue.record(inc)
        if c < job_size:
            jobs[c].inter_arrival_remaining -= inc
            if jobs[c].inter_arrival_remaining == 0:
                scheduler.queue.append(jobs[c])
                jobs[c].arrival_time = t
                c += 1
                pbar.update(1)
                arrived = True
        sch_queue_update(scheduler.queue, inc, jobs[c - 1], arrived)
        sent, server_to_send = sch_core_update(scheduler, inc, servers)
        for i in range(len(servers)):
            server_queue_update(servers[i].queue, inc, sent)
            server_cores_update(servers[i], inc, t)
    pbar.close()
    return t


print("Enter λ, α, μ:")
lam, alpha, mu = map(float, input().split())
servers = []
print("Enter means for cores of the servers:")
for i in range(5):
    means = [float(j) for j in input().split()]
    servers.append(Server(means))
job_size = 1000000
jobs = jobs_generate()
scheduler = Scheduler(mu)
final_time = simulate(jobs, scheduler, servers)
headers = ["Parameters", "Type 1", "Type 2", "Total"]
total_time = ["Total time spent in the system",
              round(np.average([j.finish_time - j.arrival_time for j in jobs if (j.type == 1 and j.processed)]), 4),
              round(np.average([j.finish_time - j.arrival_time for j in jobs if (j.type == 2 and j.processed)]), 4),
              round(np.average([j.finish_time - j.arrival_time for j in jobs if j.processed]), 4)]
queue_time = ["Time spent in queue",
              round(np.average(
                  [j.time_limit - (j.time_remaining + j.sch_time) for j in jobs if (j.type == 1 and j.processed)]), 4),
              round(np.average(
                  [j.time_limit - (j.time_remaining + j.sch_time) for j in jobs if (j.type == 2 and j.processed)]), 4),
              round(np.average([j.time_limit - (j.time_remaining + j.sch_time) for j in jobs if j.processed]), 4)]
time_limit = ["Percent of expired jobs",
              round(len([j for j in jobs if (j.type == 1 and (not j.processed))]) / len(
                  [j for j in jobs if j.type == 1]) * 100, 4),
              round(len([j for j in jobs if (j.type == 2 and (not j.processed))]) / len(
                  [j for j in jobs if j.type == 2]) * 100, 4),
              round(len([j for j in jobs if not j.processed]) / len(jobs) * 100, 4)]

tab = [total_time, queue_time, time_limit]
print(tabulate(tab, headers=headers, tablefmt='grid', numalign='center', stralign='center'))

headers2 = ["Parameter", "Scheduler", "Server 1", "Server 2", "Server 3", "Server 4", "Server 5"]
queue_length_avg = ["Avg. Queue Length", round(queue_avg(scheduler.queue, final_time), 4)]
for i in servers:
    queue_length_avg.append(round(queue_avg(i.queue, final_time), 4))
print(tabulate([queue_length_avg], headers=headers2, tablefmt='grid', numalign='center', stralign='center'))
