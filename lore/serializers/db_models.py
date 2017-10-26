import lore.env

if lore.env.jardin_conf:
  import jardin
else:
  raise ImportError('config/jardin_conf.py does not exist.')


class LoreModel(jardin.Model):
  db_names = {'master': 'master', 'replica': 'replica'}


class PredictionModels(LoreModel): pass

class PredictionModelVariants(LoreModel): pass

class PredictionModelTrainingRuns(LoreModel): pass

class FeatureSnapshotTrainingRuns(LoreModel): pass
