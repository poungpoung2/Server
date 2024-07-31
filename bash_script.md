# Bash 
Put `#!<PATH_TO_BASH>` at the start of script
eg `#!/bin/bash`
## Variabls and output
**Variables**
`VAR_NAME=VALUE` (eg `age=20`)
- When creating variables, there is **NO SPACING**.
- Commom practice to make the variable name in **small caps** when it is not an environment variable (Check system variables using `printenv`.

**Output**
`echo "<print_statement> $<VAR_NAME>"`# When outputing variable name with a print statement

## Math operation
`expr <MATH_OPP>` # When computing mathematic operations
If doing multiplication, use `\*`.
## Subcommands
When want to use commands in the script:
`$(<COMMAND>)`
### Examples
`time=$(date)` # Stores the date in the time variable
`sum=$(expr $n1 + $n2)` # Stores the sum of two variables
## If and else statement:
**Syntax**:
```
if [ <Condition> ]
then
    <Command>
else`
fi
```
**(The spaces does matter )**
### Conditions
`-eq`: equals		`-nq`: not equals
`-lt` less than `gt`: greater than
`!`: not
### Examples
```
if [ $num -gt 200 ]`
then`
    echo "$num is greater 200"`
else
	echo "num is not greater 200"
```
### Useful Examples 
**Check if the file exits**
```
if [ -f ~/myfile ]`
then`
    echo "Myfile exists"
else
	echo "Myfile doesn't exist"
```
`touch <file_na>` # Create the file
`which <command_name>` # Returns the path of the given command
**Check if the command exists**
```
command=htop
if command -v $command
then
    echo "Command is available"
else
	echo "Command is not available"
	sudo apt update && sudo apt install -y command
fi
$command 
```
`command -v` verifies if the the command exists
**[ ] in if statements**
[] is an alias for the `test` command Statemetns without [] directly evalutes the commands.
`[]`
- For basic condition checks
- String, numerical comparisons, file checks

No `[]`
- When dealing with commands
## Exit Codes
`$?` # Holds the exit status of the last executed command
(0: **success**)

### Example
```
package=htop

sudo apt install $package >> package_installation_results.log

if [ $? -eq 0 ]
then
    echo "Installation Success"
    echo "The command is available: "
    which $package
else
	echo "Installation failed" >> package_install_failture.log
fi
```
`<COMMAND> >> <FILE_NAME>` #appends the output to a file

`exit <VAL>` # Set the value for the exit code

```
dir=/etc

if [ -d $dir ]
then
	echo "Dir exists"
	exit 0
else
	echo "Doesn't exist'
	exit 1
fi
```

## Universal Update Code

```
release_file = /etc/os-release
if grep -q "Arch" $release_file
then
	sudo pacman -Syu
fi

if grep -q "Debian" $release_file || grep -q "Ubuntu" $release_file
then
	sudo apt update
	sudo apt dist-upgrade
fi
```

## For Loop
```
for <Var_Name> in <Range>
do
	<Computations>
done
```
Range: `{1..10}` # 1 ~ 10
 
### Useful Example
** Create tarballs for log files **
```
for file in logfiles/*.log
do
	tar -czvf $file.tar.gz $file
done
```

## Storing scripts

`mv <Script> /usr/local/bin` # Move the script to the bin
`sudo chown root:root <PATH_TO_SCRIPT>` # Change the user of the script to root so it is more accesible 

** Setting up path**
`export PATH=<PATH>:$PATH`

## Data Streams

`find /etc -type -f 2> /dev/null` # Any standard error would not be shown and sent to null 
(Exit code for standard error is 2)
`find /etc -type -f > /dev/null` # Send all standard output to null (putting 1 is optional)
(Exit code for standard output is 1)
`find /etc -type f & > file.txt` # Send both standard output and error
`>` and `>>`
- `>` overwrites whereas `>>` append new line

`2>&1` # Send the error to the same place as the output 

###  Examples
`find /etc -type f 1>find_results.txt 2>find_errors.txt` # Seperate error and output messages and save them in a text file
 
 **Updated update script**
```
release_file = /etc/os-release
log_file = /var/log/updater.log
errorlog = /var/log/updater_errors.log
if grep -q "Arch" $release_file
then
	sudo pacman -Syu 1>>$logfile 2>>$errorlog
	if [ $? -ne 0 ]
	then
		echo "An error occured, check $errorlog file"
	fi
fi

if grep -q "Debian" $release_file || grep -q "Ubuntu" $release_file
then
	sudo apt update 1>>$logfile 2>>$errorlog

	if [ $? -ne 0 ]
	then
		echo "An error occured, check $errorlog file"
	fi
	sudo apt dist-upgrade -y 1>>$logfile 2>>$errorlog
	if [ $? -ne 0 ]
	then
		echo "An error occured, check $errorlog file"
	fi
fi
```

## Fuctions

```
function_name (){
	<COMMAND>
}
```

### Example
**Updated update script**

```
check_exit_status(){
	if [ $? -ne 0 ]
	then
		echo "An error occured, check $errorlog file"
	fi
}


release_file = /etc/os-release
log_file = /var/log/updater.log
errorlog = /var/log/updater_errors.log
if grep -q "Arch" $release_file
then
	sudo pacman -Syu 1>>$logfile 2>>$errorlog
	check_exit_status
fi

if grep -q "Debian" $release_file || grep -q "Ubuntu" $release_file
then
	sudo apt update 1>>$logfile 2>>$errorlog
	check_exit_status
	sudo apt dist-upgrade -y 1>>$logfile 2>>$errorlog
	check_exit_status
fi 
```

## Case Statements

```
case <Var_Name> in
	pattern) 
		# commands 
		;;
	...
	*) 
		# default commands
		;;
esac

```


## Scheduling Jobs

### At
`at <Time> -f <SCRIPT_TO_LEARN>` # Scheduling a job
`atq` # Check job queue
`atrm <Queue number>` #Cancel the job

### crontab

`crontab -e` # Open editor
`30 1 * * 5 /user/local/bin.script` # <Mintue> <Hour> <Day_of_Month> <Day_of_week> <path_to_script>   

## Arguments

`<Command> $1` # Use the first argument 
`$#` # the number of arguments that are passed
 

## Backup Script

```
#!/bin/bash

# Check if the correct number of arguments were provided
if [ $# -ne 2 ]; then
    echo "Wrong number of arguments" 
    echo "./backup.sh <source_dir> <target_dir>"
    exit 1
fi

# Check if rsync is installed
if ! command -v rsync > /dev/null 2>&1; then
    echo "Rsync not installed"
    exit 2
fi

# Get the current date in the format YYYY-MM-DD
current_date=$(date +%Y-%m-%d)

# Set rsync options
rsync_options="-avb --backup-dir=$2/$current_date --delete --dry-run"

# Perform the rsync operation and log the output
$(which rsync) $rsync_options $1 $2/current >> backup_$current_date.log
```