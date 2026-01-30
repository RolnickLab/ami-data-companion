"""Pipeline registration with Antenna projects."""

import socket

import requests

from trapdata.antenna.schemas import (
    AsyncPipelineRegistrationRequest,
    AsyncPipelineRegistrationResponse,
)
from trapdata.api.api import CLASSIFIER_CHOICES, initialize_service_info
from trapdata.api.utils import get_http_session
from trapdata.common.logs import logger
from trapdata.settings import Settings, read_settings


def register_pipelines_for_project(
    base_url: str,
    auth_token: str,
    project_id: int,
    service_name: str,
    pipeline_configs: list,
) -> tuple[bool, str]:
    """
    Register all available pipelines for a specific project.

    Args:
        base_url: Base URL for the API (should NOT include /api/v2)
        auth_token: API authentication token
        project_id: Project ID to register pipelines for
        service_name: Name of the processing service
        pipeline_configs: Pre-built pipeline configuration objects

    Returns:
        Tuple of (success: bool, message: str)
    """
    with get_http_session(auth_token=auth_token) as session:
        try:
            registration_request = AsyncPipelineRegistrationRequest(
                processing_service_name=service_name, pipelines=pipeline_configs
            )

            url = f"{base_url.rstrip('/')}/projects/{project_id}/pipelines/"
            response = session.post(
                url,
                json=registration_request.model_dump(mode="json"),
                timeout=60,
            )
            response.raise_for_status()

            result = AsyncPipelineRegistrationResponse.model_validate(response.json())
            return True, f"Created {len(result.pipelines_created)} new pipelines"

        except requests.RequestException as e:
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == 400
            ):
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("detail", str(e))
                except Exception:
                    error_detail = str(e)
                return False, f"Registration failed: {error_detail}"
            else:
                return False, f"Network error during registration: {e}"
        except Exception as e:
            return False, f"Unexpected error during registration: {e}"


def register_pipelines(
    project_ids: list[int],
    service_name: str,
    settings: Settings | None = None,
) -> None:
    """
    Register pipelines for specified projects or all accessible projects.

    Args:
        project_ids: List of specific project IDs to register for. If empty, registers for all accessible projects.
        service_name: Name of the processing service
        settings: Settings object with antenna_api_* configuration (defaults to read_settings())
    """
    # Import here to avoid circular import
    from trapdata.antenna.client import get_user_projects

    # Get settings from parameter or read from environment
    if settings is None:
        settings = read_settings()

    base_url = settings.antenna_api_base_url
    auth_token = settings.antenna_api_auth_token

    if not auth_token:
        logger.error("AMI_ANTENNA_API_AUTH_TOKEN environment variable not set")
        return

    if service_name is None:
        logger.error("Service name is required for registration")
        return

    # Add hostname to service name
    hostname = socket.gethostname()
    full_service_name = f"{service_name} ({hostname})"

    # Get projects to register for
    projects_to_process = []
    if project_ids:
        # Use specified project IDs
        projects_to_process = [
            {"id": pid, "name": f"Project {pid}"} for pid in project_ids
        ]
        logger.info(f"Registering pipelines for specified projects: {project_ids}")
    else:
        # Fetch all accessible projects
        logger.info("Fetching all accessible projects...")
        all_projects = get_user_projects(base_url, auth_token)
        projects_to_process = all_projects
        logger.info(f"Found {len(projects_to_process)} accessible projects")

    if not projects_to_process:
        logger.warning("No projects found to register pipelines for")
        return

    # Initialize service info once to get pipeline configurations
    logger.info("Initializing pipeline configurations...")
    service_info = initialize_service_info()
    pipeline_configs = service_info.pipelines
    logger.info(f"Generated {len(pipeline_configs)} pipeline configurations")

    # Register pipelines for each project
    successful_registrations = []
    failed_registrations = []

    logger.info(f"Available pipelines to register: {list(CLASSIFIER_CHOICES.keys())}")

    for project in projects_to_process:
        project_id = project["id"]
        project_name = project.get("name", f"Project {project_id}")

        logger.info(
            f"Registering pipelines for project {project_id} ({project_name})..."
        )

        success, message = register_pipelines_for_project(
            base_url=base_url,
            auth_token=auth_token,
            project_id=project_id,
            service_name=full_service_name,
            pipeline_configs=pipeline_configs,
        )

        if success:
            successful_registrations.append((project_id, project_name, message))
            logger.info(f"✓ Project {project_id} ({project_name}): {message}")
        else:
            failed_registrations.append((project_id, project_name, message))
            if "Processing service already exists" in message:
                logger.warning(f"⚠ Project {project_id} ({project_name}): {message}")
            else:
                logger.error(f"✗ Project {project_id} ({project_name}): {message}")

    # Summary report
    logger.info("\n=== Registration Summary ===")
    logger.info(f"Service name: {full_service_name}")
    logger.info(f"Total projects processed: {len(projects_to_process)}")
    logger.info(f"Successful registrations: {len(successful_registrations)}")
    logger.info(f"Failed registrations: {len(failed_registrations)}")

    if successful_registrations:
        logger.info("\nSuccessful registrations:")
        for project_id, project_name, message in successful_registrations:
            logger.info(f"  - Project {project_id} ({project_name}): {message}")

    if failed_registrations:
        logger.info("\nFailed registrations:")
        for project_id, project_name, message in failed_registrations:
            logger.info(f"  - Project {project_id} ({project_name}): {message}")
