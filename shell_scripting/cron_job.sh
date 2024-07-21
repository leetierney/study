#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the logfile path based on the script's directory
LOGFILE="${SCRIPT_DIR}/logfile.log"

# Define the cron job with the dynamic logfile path, set to run every minute
CRON_JOB="* * * * * date '+\%Y-\%m-\%d \%H:\%M:\%S' >> ${LOGFILE}"

# Check if the cron job already exists
(crontab -l | grep -F "$CRON_JOB") || (crontab -l; echo "$CRON_JOB") | crontab -
