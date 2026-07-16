"""Unit tests for model staging and deployment planning helpers."""

from __future__ import annotations

import sys
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock


SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / ".kiro"
    / "skills"
    / "sagemaker-deploy"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))

import deploy  # noqa: E402
import stage_model  # noqa: E402
import config  # noqa: E402


class DeployHelperTests(unittest.TestCase):
    def test_build_vllm_env(self):
        env = deploy.build_env(
            "vllm",
            8,
            65535,
            8,
            {
                "SM_VLLM_ENABLE_EXPERT_PARALLEL": "true",
                "SM_VLLM_GPU_MEMORY_UTILIZATION": 0.9,
            },
        )
        self.assertEqual("8", env["SM_VLLM_TENSOR_PARALLEL_SIZE"])
        self.assertEqual("65535", env["SM_VLLM_MAX_MODEL_LEN"])
        self.assertEqual("0.9", env["SM_VLLM_GPU_MEMORY_UTILIZATION"])

    def test_build_sglang_env(self):
        env = deploy.build_env(
            "sglang",
            8,
            65535,
            None,
            {"SM_SGLANG_QUANTIZATION": "modelopt_fp4"},
        )
        self.assertEqual("/opt/ml/model", env["SM_SGLANG_MODEL_PATH"])
        self.assertEqual("8", env["SM_SGLANG_TP"])
        self.assertEqual("modelopt_fp4", env["SM_SGLANG_QUANTIZATION"])

    def test_build_env_normalizes_boolean_flags_and_protects_topology(self):
        env = deploy.build_env(
            "vllm",
            8,
            None,
            None,
            {
                "SM_VLLM_ENABLE_EXPERT_PARALLEL": True,
                "SM_VLLM_ENFORCE_EAGER": False,
            },
        )
        self.assertEqual("", env["SM_VLLM_ENABLE_EXPERT_PARALLEL"])
        self.assertNotIn("SM_VLLM_ENFORCE_EAGER", env)
        with self.assertRaisesRegex(ValueError, "cannot override"):
            deploy.build_env(
                "vllm",
                8,
                None,
                None,
                {"SM_VLLM_TENSOR_PARALLEL_SIZE": "4"},
            )

    def test_build_env_rejects_null(self):
        with self.assertRaisesRegex(ValueError, "cannot be null"):
            deploy.build_env(
                "sglang",
                1,
                None,
                None,
                {"SM_SGLANG_REASONING_PARSER": None},
            )

    def test_cuda_compatibility(self):
        self.assertTrue(deploy.cuda_compatible("ml.g7e.48xlarge", 130))
        self.assertFalse(deploy.cuda_compatible("ml.g7e.48xlarge", 129))
        self.assertTrue(deploy.cuda_compatible("ml.g5.12xlarge", 128))
        self.assertFalse(deploy.cuda_compatible("ml.g5.12xlarge", 130))

    def test_resource_name_obeys_sagemaker_limit(self):
        name = deploy.resource_name("x" * 100, "sglang", 2)
        self.assertLessEqual(len(name), 63)
        self.assertRegex(name, r"^[A-Za-z0-9-]+$")

    def test_s3_inspection_requires_and_verifies_manifest(self):
        manifest = {
            "repo_id": "org/model",
            "revision": "abc123",
            "files": [
                {"path": "config.json", "size": 10},
                {"path": "model.safetensors", "size": 20},
            ],
        }
        s3 = mock.Mock()
        paginator = s3.get_paginator.return_value
        paginator.paginate.return_value = [{
            "Contents": [
                {"Key": "models/test/config.json", "Size": 10},
                {"Key": "models/test/model.safetensors", "Size": 20},
                {
                    "Key": "models/test/.hf-model-manifest.json",
                    "Size": 100,
                },
            ],
        }]
        s3.get_object.return_value = {
            "Body": BytesIO(stage_model.json.dumps(manifest).encode()),
        }

        summary = deploy.inspect_s3_model(s3, "s3://bucket/models/test/")

        self.assertEqual("org/model", summary["source_model"])
        self.assertEqual("abc123", summary["revision"])

    def test_s3_inspection_rejects_missing_manifest(self):
        s3 = mock.Mock()
        paginator = s3.get_paginator.return_value
        paginator.paginate.return_value = [{
            "Contents": [
                {"Key": "models/test/config.json", "Size": 10},
                {"Key": "models/test/model.safetensors", "Size": 20},
            ],
        }]
        with self.assertRaisesRegex(RuntimeError, "manifest"):
            deploy.inspect_s3_model(s3, "s3://bucket/models/test/")


class StagingHelperTests(unittest.TestCase):
    def test_default_destination_uses_full_repository_id(self):
        with mock.patch.object(stage_model.config, "bucket", return_value="bucket"):
            self.assertEqual(
                "s3://bucket/models/organization--model-name/",
                stage_model.default_destination("Organization/Model Name"),
            )

    def test_parse_s3_uri(self):
        self.assertEqual(
            ("example", "models/model/"),
            stage_model.parse_s3_uri("s3://example/models/model"),
        )
        with self.assertRaises(ValueError):
            stage_model.parse_s3_uri("https://example/model")

    def test_source_url_pins_revision_and_escapes_path(self):
        url = stage_model.source_url("org/model", "abc123", "nested/a file.json")
        self.assertIn("/org/model/resolve/abc123/", url)
        self.assertIn("nested/a%20file.json", url)

    def test_open_hf_url_rejects_unsafe_targets(self):
        for url in (
            "file:///tmp/model.json",
            "http://huggingface.co/api/models/org/model",
            "https://huggingface.co.evil.example/api/models/org/model",
            "https://user:password@huggingface.co/api/models/org/model",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                stage_model.open_hf_url(url, timeout=60)

    def test_open_hf_url_uses_validated_https_opener(self):
        opener = mock.Mock()
        response = mock.Mock()
        opener.open.return_value = response
        url = "https://huggingface.co/api/models/org/model"

        with mock.patch.object(
            stage_model.urllib.request,
            "build_opener",
            return_value=opener,
        ):
            actual = stage_model.open_hf_url(url, timeout=60)

        self.assertIs(response, actual)
        request = opener.open.call_args.args[0]
        self.assertEqual(url, request.full_url)
        opener.open.assert_called_once_with(request, timeout=60)

    def test_redirect_handler_rejects_non_https_target(self):
        handler = stage_model.HttpsOnlyRedirectHandler()
        with self.assertRaises(ValueError):
            handler.redirect_request(
                None,
                None,
                302,
                "Found",
                {},
                "file:///tmp/model.safetensors",
            )

    def test_load_hf_token_supports_json_secret(self):
        secrets = mock.Mock()
        secrets.get_secret_value.return_value = {
            "SecretString": '{"HF_TOKEN":"secret-value"}'
        }
        with mock.patch.object(
            stage_model.boto3,
            "client",
            return_value=secrets,
        ), mock.patch.dict(stage_model.os.environ, {}, clear=True):
            stage_model.load_hf_token("hf-token")
            self.assertEqual("secret-value", stage_model.os.environ["HF_TOKEN"])


class ConfigTests(unittest.TestCase):
    def test_execution_role_discovery_rejects_ambiguous_matches(self):
        fake_sagemaker = types.ModuleType("sagemaker")
        fake_sagemaker.session = types.SimpleNamespace(
            Session=mock.Mock(side_effect=RuntimeError("outside Studio"))
        )
        sts = mock.Mock()
        sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/test"
        }
        iam = mock.Mock()
        iam.get_paginator.return_value.paginate.return_value = [{
            "Roles": [
                {
                    "RoleName": "AmazonSageMaker-ExecutionRole-A",
                    "Arn": "arn:aws:iam::123456789012:role/A",
                },
                {
                    "RoleName": "AmazonSageMaker-ExecutionRole-B",
                    "Arn": "arn:aws:iam::123456789012:role/B",
                },
            ]
        }]
        session = mock.Mock()
        session.client.side_effect = lambda service: {
            "sts": sts,
            "iam": iam,
        }[service]

        with mock.patch.dict(config.os.environ, {}, clear=True), mock.patch.dict(
            sys.modules,
            {"sagemaker": fake_sagemaker},
        ):
            with self.assertRaisesRegex(RuntimeError, "Multiple SageMaker"):
                config.execution_role_arn(session)


if __name__ == "__main__":
    unittest.main()
