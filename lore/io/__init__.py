import os
import tempfile
import re
import configparser
import sys
import logging
import redis

import lore
from lore.util import timer
from lore.io.connection import Connection

if not (sys.version_info.major == 3 and sys.version_info.minor >= 6):
    ModuleNotFoundError = ImportError
try:
    import boto3
    from botocore.exceptions import ClientError
except ModuleNotFoundError as e:
    boto3 = False
    ClientError = Exception

logger = logging.getLogger(__name__)


config = lore.env.database_config
if config:
    try:
        for database, url in config.items('DATABASES'):
            vars()[database] = Connection(url=url)
    except configparser.NoSectionError:
        pass

    for section in config.sections():
        if section == 'DATABASES':
            continue
            
        options = config._sections[section]
        vars()[section.lower()] = Connection(**options)

redis_config = lore.env.redis_config

if redis_config:
    try:
        for section in config.sections():
            vars()[section.lower()] = redis.StrictRedis(host=redis_config.get(section, 'url'),
                                                        port=redis_config.get(section, 'port'))
    except:
        pass
else:
    redis_conn = redis.StrictRedis(host='localhost', port=6379)

if boto3:
    config = lore.env.aws_config
    s3 = None
    if config and 'ACCESS_KEY' in config.sections():
        s3 = boto3.resource(
            's3',
            aws_access_key_id=config.get('ACCESS_KEY', 'id'),
            aws_secret_access_key=config.get('ACCESS_KEY', 'secret')
        )
    else:
        s3 = boto3.resource('s3')

    if s3 and config and 'BUCKET' in config.sections():
        bucket = s3.Bucket(config.get('BUCKET', 'name'))

def download(local_path, remote_path=None, cache=True):
    if not remote_path:
        remote_path = local_path
        remote_path = re.sub(
            r'^%s' % re.escape(lore.env.work_dir),
            lore.env.name,
            remote_path
        )

    if not remote_path.startswith(lore.env.name):
        remote_path = os.path.join(lore.env.name, remote_path)

    if cache and os.path.exists(local_path):
        return

    dir = os.path.dirname(local_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    with timer('DOWNLOAD: %s -> %s' % (remote_path, local_path)):
        temp_file, temp_path = tempfile.mkstemp()
        try:
            bucket.download_file(remote_path, temp_path)
        except ClientError as e:
            logger.error("Error downloading file: %s" % e)
            raise
        
        os.rename(temp_path, local_path)


def upload(local_path, remote_path=None):
    if remote_path:
        if remote_path.startswith('/'):
            logger.warning('remotepath should be relative (to bucket root)')
            remote_path = remote_path[1:-1]
        if not remote_path.startswith(lore.env.name):
            remote_path = os.path.join(lore.env.name, remote_path)
    else:
        remote_path = local_path
        remote_path = re.sub(
            r'^%s' % re.escape(lore.env.work_dir),
            lore.env.name,
            remote_path
        )

    with timer('UPLOAD: %s -> %s' % (local_path, remote_path)):
        try:
            bucket.upload_file(local_path, remote_path, ExtraArgs={'ServerSideEncryption': 'AES256'})
        except ClientError as e:
            logger.error("Error uploading file: %s" % e)
            raise
