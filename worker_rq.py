# from redis import Redis
# from rq import Worker, Queue

# def worker_main():
#     # Connect to Redis server
#     redis_conn = Redis(host='localhost', port=6379, db=0)

#     # Create a new Queue
#     queue = Queue(connection=redis_conn)

#     # Start a worker to process jobs from the new queue
#     worker = Worker([queue], connection=redis_conn)
#     worker.work()

# if __name__ == '__main__':
#     worker_main()

from redis import Redis
from rq import Worker, Queue, Connection
import logging

def worker_main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Connect to Redis server
    redis_conn = Redis(host='localhost', port=6379, db=0)
    
    # Create a new Queue
    queue = Queue(connection=redis_conn)
    
    # Define a custom worker class to print job results
    class CustomWorker(Worker):
        def execute_job(self, job, queue):
            logging.info(f"Starting job: {job.id}")
            result = super().execute_job(job, queue)
            logging.info(f"Finished job: {job.id} with result: {job.result}")
            return result

    # Start a worker to process jobs from the new queue
    with Connection(redis_conn):
        worker = CustomWorker([queue])
        worker.work(with_scheduler=True)  # Start the worker with the scheduler

if __name__ == '__main__':
    worker_main()