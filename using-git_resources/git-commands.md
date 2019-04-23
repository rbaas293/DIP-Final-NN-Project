### Create a new Local branch based on your current checked out branch
```Shell
git checkout -b <new-branch-name>
```
* At this point you would edit/add files to your new branch and then commit the changes.
### Push your branch to the remote repository:
```Shell
git push -u origin <new-branch-name>
```
### Delete Local then Remote Branch 
```Shell
git checkout <not-branch-to-be-deleted>
git branch -d <branch-to-delete>
git push origin --delete  <branch-to-delete>
```

### Ignoring a previously committed file
*  If you want to ignore a file that you've committed in the past, you'll need to delete the file from your repository and then add a `.gitignore` rule for it. Using the `--cached` option with git rm means that the file will be deleted from your repository, but will remain in your working directory as an ignored file.
```Shell
echo debug.log >> .gitignore      # Add the rule
git rm --cached debug.log
git commit -m "Start ignoring debug.log"
```
* You can omit the --cached option if you want to delete the file from both the repository and your local file system.
