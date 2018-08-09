import lore.spark
from lore.tasks.base import Base

class Spark(Base):
    def main(self, **kwargs):
        lore.spark.context
