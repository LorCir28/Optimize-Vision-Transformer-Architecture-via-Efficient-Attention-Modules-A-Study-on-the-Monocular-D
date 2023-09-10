--- How run tests 
1) Go on the folder relative to the METER variant that you want to use
2) Open terminal
3) Type "python METER_VARIANT_NAME.py --os_type OS_TYPE --run RUN"
	-> METER_VARIAMT_NAME = the name of the only Python file in the folder
	-> OS_TYPE = 'w' if you're using Windows, 'l' if you're using Linux or Ubuntu (MacOS? Bleah)
	-> RUN = 'trial' to make 1 run on only xxs architecture, 'real' to make 30 runs on s,xs,xxs architectures