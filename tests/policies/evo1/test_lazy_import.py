"""Configuration discovery must not import the optional EVO1 model."""

import subprocess
import sys
import textwrap


def test_evo1_configuration_does_not_load_model():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import lerobot.policies.evo1 as package; "
            "import sys; "
            "assert package.Evo1Config.__name__ == 'Evo1Config'; "
            "assert 'lerobot.policies.evo1.modeling_evo1' not in sys.modules; "
            "assert 'lerobot.policies.evo1.internvl3_embedder' not in sys.modules",
        ],
        check=True,
    )


def test_remote_record_import_does_not_require_model_or_piper_packages():
    script = textwrap.dedent("""
        import builtins
        import importlib.util

        unavailable = {"transformers", "piper_sdk", "pinocchio"}
        original_find_spec = importlib.util.find_spec
        original_import = builtins.__import__

        def find_spec(name, *args, **kwargs):
            if name.split(".")[0] in unavailable:
                return None
            return original_find_spec(name, *args, **kwargs)

        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in unavailable:
                raise ModuleNotFoundError(name, name=name)
            return original_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        builtins.__import__ = checked_import

        import lerobot.scripts.lerobot_rlt_record
        import lerobot.async_inference.helpers
        import lerobot.transport.services_pb2_grpc
    """)
    subprocess.run([sys.executable, "-c", script], check=True)
