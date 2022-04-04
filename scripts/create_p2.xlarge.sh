#!/bin/bash

aws ec2 run-instances --image-id $1 --count 1 --instance-type p2.xlarge --key-name aakashAndVivek --security-group-ids sg-f80c3cab --region us-east-1 --private-ip-address 172.31.65.5 --tag-specifications 'ResourceType=instance, Tags=[{Key=Name, Value=slave0}, {Key=hostname, Value=ip-172-31-65-5}]'
