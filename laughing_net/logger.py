import neptune
import dvclive

from laughing_net.config import config
from laughing_net.utils.git import get_current_branch

class Logger:
    def __init__(self,
        tags = None,
        dry_run = False,
        **kwargs,
    ):
        self.api_token = config.neptune.api_token
        self.project_name = config.neptune.project_name
        self.tags = tags or list()
        self.dry_run = dry_run
        self.kwargs = kwargs
        self.create_experiment()

    def dvclive_next_step(self):
        dvclive.next_step()

    def log_metric(self, log_name, x, y=None, timestamp=None, dvc=False):
        if dvc:
            if y is not None:
                dvclive.log(log_name, y, x)
            else:
                dvclive.log(log_name, x)
        self.experiment.log_metric(log_name, x, y, timestamp)

    def log_text(self, log_name, x, y=None, timestamp=None):
        self.experiment.log_text(log_name, x, y, timestamp)

    def log_image(self, log_name, x, y=None, image_name=None, description=None, timestamp=None):
        self.experiment.log_image(log_name, x, y, image_name, description, timestamp)

    def log_artifact(self, artifact, destination=None):
        self.experiment.log_artifact(artifact, destination)

    def create_experiment(self):
        # dvclive.init()

        if self.dry_run:
            neptune.init(
                project_qualified_name="dry-run/debug",
                backend=neptune.OfflineBackend()
            )
        else:
            neptune.init(
                api_token=self.api_token,
                project_qualified_name=self.project_name,
            )

        self.tags = list({get_current_branch()} | set(self.tags))

        self.experiment = neptune.create_experiment(
            tags=self.tags,
            **self.kwargs,
        )

logger = object.__new__(Logger)
init_logger = logger.__init__
