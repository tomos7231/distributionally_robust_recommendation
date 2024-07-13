# distributionally-robust-recommendation

## Setup
This repository is using rye.  
If you want to use this repository, please run the following command.

1. install rye
   - install instructions: https://rye-up.com/guide/installation/
2. enable uv to speed up dependency resolution.
```
rye config --set-bool behavior.use-uv=true
```
3. create a virtual environment
```
rye sync
```

