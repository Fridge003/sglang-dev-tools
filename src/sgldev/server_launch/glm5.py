"""GLM-5 launch command."""

from sgldev.server_launch._base import make_launch_command

glm5 = make_launch_command("zai-org/GLM-5-FP8")
glm5.__name__ = "glm5"
glm5.__doc__ = """\
Launch GLM-5.

Combine --dp and --mtp to select a configuration:

\b
    sgldev server launch glm5              # TP8 (default)
    sgldev server launch glm5 --dp         # DP8
    sgldev server launch glm5 --mtp        # TP8 + MTP
    sgldev server launch glm5 --dp --mtp   # DP8 + MTP
"""
