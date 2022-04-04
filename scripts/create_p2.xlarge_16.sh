#!/bin/bash

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.5 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave0-p2xlarge}, {Key=hostname, Value=ip-172-31-65-5}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.6 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave1-p2xlarge}, {Key=hostname, Value=ip-172-31-65-6}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.12 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave2-p2xlarge}, {Key=hostname, Value=ip-172-31-65-12}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.8 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave3-p2xlarge}, {Key=hostname, Value=ip-172-31-65-8}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.9 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave4-p2xlarge}, {Key=hostname, Value=ip-172-31-65-9}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.10 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave5-p2xlarge}, {Key=hostname, Value=ip-172-31-65-10}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.11 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave6-p2xlarge}, {Key=hostname, Value=ip-172-31-65-11}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.13 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave7-p2xlarge}, {Key=hostname, Value=ip-172-31-65-13}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.14 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave8-p2xlarge}, {Key=hostname, Value=ip-172-31-65-14}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.15 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave9-p2xlarge}, {Key=hostname, Value=ip-172-31-65-15}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.16 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave10-p2xlarge}, {Key=hostname, Value=ip-172-31-65-16}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.17 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave11-p2xlarge}, {Key=hostname, Value=ip-172-31-65-17}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.18 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave12-p2xlarge}, {Key=hostname, Value=ip-172-31-65-18}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.19 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave13-p2xlarge}, {Key=hostname, Value=ip-172-31-65-19}]'

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.20 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave14-p2xlarge}, {Key=hostname, Value=ip-172-31-65-20}]'


