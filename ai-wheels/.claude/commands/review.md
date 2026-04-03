Please review a given SGLang pull requests passed in by the slash command. It's in the form of `/review x`, where x is the pull request number.

For example, if I execute `/review 19080`, this command should go to the correct PR link: `https://github.com/sgl-project/sglang/pull/19089`, create a new branch and pull this PR change with github CLI, and then review the diffs vs main branch. After reviewing, please delete the reviewed branch locally.

During reviewing, please ranks all the potential risks from high to los. 
Remember to check the following points:
- It's always dangerous to modify any file under `python/sglang/srt/managers`. If so, raise a highest risk level
- Make sure the user is adding according CI tests if he/she is developing a feature rather than fixing a bug
- If the user is appending new server arguments in `python/sglang/srt/server_args.py`, he should also update document `docs/advanced_features/server_arguments.md`.
- If the user is appending new environ in `python/sglang/srt/environ.py`, he should also update document `docs/references/environment_variables.md`
- To be added
