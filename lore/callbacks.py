import lore
from lore.env import require
from lore.util import timer

import logging
from datetime import datetime

require(lore.dependencies.KERAS)
import keras.callbacks


logger = logging.getLogger(__name__)


class ReloadBest(keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        monitor='val_loss',
        mode='auto',
    ):
        super(ReloadBest, self).__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=0,
            mode=mode,
            save_best_only=False,
            save_weights_only=True,
            period=1
        )
        self.train_loss = None
        self.validate_loss = None
        self.best_epoch = None
        self.train_begin = None

    def on_train_begin(self, logs=None):
        super(ReloadBest, self).on_train_begin(logs)

        self.train_begin = datetime.utcnow()
        logger.info('=============================================')
        logger.info('|    epoch |     time |    train | validate |')
        logger.info('---------------------------------------------')

    def on_train_end(self, logs=None):
        super(ReloadBest, self).on_train_end(logs)
        logger.info('=============================================')
        if self.best_epoch is not None:
            logger.debug('best epoch: %i' % self.best_epoch)
            with timer('load best epoch'):
                self.model.load_weights(
                    self.filepath.format(epoch=self.best_epoch)
                )

    def on_epoch_end(self, epoch, logs=None):
        super(ReloadBest, self).on_epoch_end(epoch, logs)
        time = datetime.utcnow() - self.train_begin
        train_loss = logs.get('loss')
        validate_loss = logs.get('val_loss')
        if validate_loss:
            if self.validate_loss is None or self.validate_loss > validate_loss:
                self.best_epoch = epoch + 1
                self.train_loss = train_loss
                self.validate_loss = validate_loss
        else:
            logger.error('No val_loss in logs, setting to NaN')
            validate_loss = float('nan')
        logger.info('| %8i | %8s | %8.4f | %8.4f |' % (
            epoch, str(time).split('.', 2)[0], train_loss, validate_loss)
        )
