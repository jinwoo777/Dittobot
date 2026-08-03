"""FastAPI dependency adapters."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from robot_skill_system.application import MVPApplication


def get_service(request: Request) -> MVPApplication:
    """Return the application service bound at app creation time."""

    return cast(MVPApplication, request.app.state.service)


ServiceDependency = Annotated[MVPApplication, Depends(get_service)]
