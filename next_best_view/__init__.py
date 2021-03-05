import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


register(id="nextbestview-v0",
         entry_point="next_best_view.envs:NextBestViewEnv"
         # kwargs={"render": True}
         )
