"""
Test-session setup shared by every test in the repository.

The tests assume the service offers every pipeline. The AMI_PIPELINES setting limits
that, and a developer may have it set in the environment or in .env, as the README
suggests for a deployment. The API's request schema is built from it when the API
module is first imported, so it is cleared here, before any test module imports that
module. An empty value in the environment also overrides a value in .env. Tests that
need a particular value set it themselves.

The API runs the BioCLIP extractor by default, which would download a ViT-H backbone
of several GB. Tests turn it off the same way and enable it with a stand-in encoder.
"""

import os

os.environ["AMI_PIPELINES"] = ""
os.environ["AMI_EMBEDDING_EXTRACTOR"] = "none"
