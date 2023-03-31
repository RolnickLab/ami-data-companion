"""missing data for tracking

Revision ID: 1544478c3031
Revises: 90d5f6ae09ec
Create Date: 2023-03-08 21:02:14.909938

"""

# revision identifiers, used by Alembic.
revision = "1544478c3031"
down_revision = "90d5f6ae09ec"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from trapdata.db.maintenance import missing_tracking_data

    missing_tracking_data.add_missing_image_data()
    missing_tracking_data.add_missing_detection_data()


def downgrade() -> None:
    pass
