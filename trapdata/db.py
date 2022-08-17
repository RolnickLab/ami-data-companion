import sqlalchemy as sa
from sqlalchemy import orm

Base = orm.declarative_base()


class Image(Base):
    __tablename__ = "images"

    id = sa.Column(sa.Integer, primary_key=True)
    base_directory = sa.Column(sa.String(255))
    path = sa.Column(sa.String(255))
    timestamp = sa.Column(sa.DateTime(timezone=True))
    num_detected_objects = sa.Column(sa.Integer)
    last_read = sa.Column(sa.DateTime)
    last_processed = sa.Column(sa.DateTime)

    detected_objects = orm.relationship(
        "DetectedObject", back_populates="image", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"Image(path={self.path!r}, num_detected_objects={self.num_detected_objects!r})"


class DetectedObject(Base):
    __tablename__ = "detections"

    id = sa.Column(sa.Integer, primary_key=True)
    image_id = sa.Column(sa.ForeignKey("images.id"))
    bbox = sa.Column(sa.JSON)
    specific_label = sa.Column(sa.String(255))
    binary_label = sa.Column(sa.Boolean)
    last_detected = sa.Column(sa.DateTime)

    image = orm.relationship("Image", back_populates="detected_objects")

    def __repr__(self):
        return f"DetectedObject(image={self.image.path!r}, bbox={self.bbox!r})"


def init_db(location=":memory:"):
    engine = sa.create_engine(f"sqlite+pysqlite:///{location}", echo=True, future=True)
    Base.metadata.create_all(engine)
    return engine


def test_db():
    from .utils import TEST_IMAGES_BASE_PATH

    engine = init_db(location=":memory:")
    with orm.Session(engine) as session:
        image = Image(
            base_directory=TEST_IMAGES_BASE_PATH,
            path="2022-01-01/test.jpg",
        )
        session.add_all([image])
        session.commit()
        bbox = DetectedObject(
            bbox=[0, 0, 0, 0], specific_label="unknown", binary_label=True
        )
        image.detected_objects.append(bbox)
        session.commit()

        stmt = sa.select(Image).where(Image.base_directory == TEST_IMAGES_BASE_PATH)
        image = session.scalars(stmt).one()

        print(image)
        print(image.detected_objects)


if __name__ == "__main__":
    test_db()
