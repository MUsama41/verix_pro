import boto3
import time

ec2 = boto3.client('ec2')

# Instance ID of your EC2
instance_id = 'i-0706a2073f420909b'

# Define the time duration after which the instance should stop (in seconds)
shutdown_time = 8 * 60 * 60 # 3 minutes in seconds

# Wait for the specified time duration
time.sleep(shutdown_time)

# Stop the EC2 instance
ec2.stop_instances(InstanceIds=[instance_id])

print(f"EC2 instance {instance_id} is shutting down after {shutdown_time / 60} minutes.")
