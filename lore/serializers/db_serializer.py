from memoized_property import memoized_property
from datetime import datetime
import lore.env

if lore.env.jardin_conf:
    from lore.serializers.db_models import PredictionModels, PredictionModelVariants, PredictionModelTrainingRuns, LoreModel, FeatureSnapshotTrainingRuns

class DbSerializer(object):

  DEFAULT_DESCRIPTION = 'Automatically created by lore'

  def __init__(self, serializer):
    self.serializer = serializer
    self.model = self.serializer.model
    self.params = self.model.db_serialization_params

  @memoized_property
  def prediction_model_name(self):
    return self.params.get('model', {}).get('name', self.model.__module__)

  @memoized_property
  def prediction_model_variant_name(self):
    return self.params.get('variant', {}).get('name', self.model.__class__.__name__)

  @memoized_property
  def prediction_model(self):
    model = PredictionModels.find_by(values={'name': self.prediction_model_name})
    if not model:
      model = PredictionModels.insert(values={
        'name': self.prediction_model_name,
        'description': self.params.get('model', {}).get('description', self.DEFAULT_DESCRIPTION),
        'owner_id': self.params['owner_id']
        })
    return model

  @memoized_property
  def prediction_model_variant(self):
    variant = PredictionModelVariants.find_by(values={
      'name': self.prediction_model_variant_name,
      'prediction_model_id': self.prediction_model.id
      })
    if not variant:
      variant = PredictionModelVariants.insert(values={
        'prediction_model_id': self.prediction_model.id,
        'name': self.prediction_model_variant_name,
        'description': self.params.get('variant', {}).get('description', self.DEFAULT_DESCRIPTION),
        'owner_id': self.prediction_model.owner_id
        })
    return variant

  def save(self, params={}, stats={}):
    if not self.params.get('serialize', False):
      return
    with LoreModel.transaction():
      training_run = PredictionModelTrainingRuns.insert({
        'prediction_model_id': self.prediction_model.id,
        'prediction_model_variant_id': self.prediction_model_variant.id,
        'training_started_at': self.model.fit_started_at,
        'training_completed_at': self.model.fit_completed_at,
        'benchmark': stats,
        'parameters': params,
        'metadata': self.params.get('metadata', {}),
        'path': self.serializer.remote_model_path
        })
      if 'prediction_feature_snapshots' in dir(self.model.pipeline):
        for snapshot in self.model.pipeline.prediction_feature_snapshots.records():
          FeatureSnapshotTrainingRuns.insert(values={
            'prediction_model_training_run_id': training_run.id,
            'prediction_feature_snapshot_id': snapshot.id
          })
        PredictionModels.update(
          values={'updated_at': datetime.utcnow()},
          where={'id': self.prediction_model_variant.prediction_model_id})
        PredictionModelVariants.update(
          values={'updated_at': datetime.utcnow()},
          where={'id': self.prediction_model_variant_id})

  def load(self, fitting=None):
    raise NotImplementedError()
