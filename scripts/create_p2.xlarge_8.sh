#!/bin/bash

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.5 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave0-p2xlarge}, {Key=hostname, Value=ip-172-31-65-5}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.6 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave1-p2xlarge}, {Key=hostname, Value=ip-172-31-65-6}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.12 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave2-p2xlarge}, {Key=hostname, Value=ip-172-31-65-12}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.8 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave3-p2xlarge}, {Key=hostname, Value=ip-172-31-65-8}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.9 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave4-p2xlarge}, {Key=hostname, Value=ip-172-31-65-9}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.10 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave5-p2xlarge}, {Key=hostname, Value=ip-172-31-65-10}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.11 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave6-p2xlarge}, {Key=hostname, Value=ip-172-31-65-11}]'

