from hatchling.builders.hooks.plugin.interface import BuildHookInterface
import subprocess

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Run your script to generate the binary file
        subprocess.check_call(["sh", "scripts/build_guest.sh"])
