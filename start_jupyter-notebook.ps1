# start the Jupiter notebook at root repository directory 
# May need to run the folowing in order run this script. no admin needed
#Set-ExecutionPolicy -ExecutionPolicy Bypass -scope CurrentUser

# Get Present Working Directory:
$currentpwd = Get-Location;

#Start The notebook in pwd so we have acess to the entire directory:
jupyter notebook --notebook-dir=$currentpwd

