#!/bin/bash
 
if [[ "$#" -ne 4 && "$#" -ne 5 ]]
then
	echo "Error: Incorrect no. of arguments. Expected 4 or 5 arguments."
elif [ "$#" -eq 4 ]
then
	if [ "$1" -ne 1 ]
	then
		echo "Error: Incorrect no. of arguments. Expected 4 arguments only for Q1."
	else
		if [ "$4" = "a" ]
		then
			python question1_a.py "$2" "$3"
		elif [ "$4" = "b" ]
		then 
			python question1_b.py "$2" "$3"
		elif [ "$4" = "c" ]
		then 
			python question1_c.py "$2" "$3"
		elif [ "$4" = "d" ]
		then 
			python question1_d.py "$2" "$3"
		elif [ "$4" = "e" ]
		then 
			python question1_e.py "$2" "$3"
		elif [ "$4" = "f" ]
		then 
			python question1_f.py "$2" "$3"
		elif [ "$4" = "g" ]
		then 
			python question1_g.py "$2" "$3"
		else
			echo "Error: Incorrect part number for Q1. Part number can be a, b, c, d, e, f or g."
		fi
	fi
else
	if [ "$1" -ne 2 ]
	then
		echo "Error: Incorrect no. of arguments. Expected 5 arguments only for Q2."
	else
		if [ "$4" -eq 0 ]
		then
			if [ "$5" = "a" ]
			then
				python question2_1a.py "$2" "$3"
			elif [ "$5" = "b" ]
			then 
				python question2_1b.py "$2" "$3"
			elif [ "$5" = "c" ]
			then 
				python question2_1c.py "$2" "$3"
			else
				echo "Error: Incorrect part number for Q2 Binary Classification. Part number can be a, b or c."
			fi
		elif [ "$4" -eq 1 ]
		then
			if [ "$5" = "a" ]
			then
				python question2_2a.py "$2" "$3"
			elif [ "$5" = "b" ]
			then 
				python question2_2b.py "$2" "$3"
			elif [ "$5" = "c" ]
			then 
				python question2_2c.py "$2" "$3"
			elif [ "$5" = "d" ]
			then 
				python question2_2d.py "$2" "$3"
			else
				echo "Error: Incorrect part number for Q2 Multi-Class Classification. Part number can be a, b, c or d."
			fi
		else
			echo "Error: Incorrect 3rd argument for Q2. Q2 3rd argument can be 0 for binary or 1 for multi-class classification."
		fi
	fi
fi

