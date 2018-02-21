import os
import tempfile
import re
import configparser
import sys
import tarfile
import logging

if sys.version_info[0] == 2:
    from urlparse import urlparse
    from urllib import urlretrieve
else:
    from urllib.parse import urlparse
    from urllib.request import urlretrieve

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

try:
    import redis
except ModuleNotFoundError as e:
    redis = False
    

logger = logging.getLogger(__name__)


config = lore.env.database_config
if config:
    try:
        for database, url in config.items('DATABASES'):
            vars()[database] = Connection(url=url, name=database)
    except configparser.NoSectionError:
        pass

    for section in config.sections():
        if section == 'DATABASES':
            continue
            
        options = config._sections[section]
        if options.get('url') == '$DATABASE_URL':
            logger.error('$DATABASE_URL is not set, but is used in config/database.cfg. Skipping connection.')
        else:
            vars()[section.lower()] = Connection(name=section.lower(), **options)

redis_config = lore.env.redis_config

if redis:
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


def download(remote_url, local_path=None, cache=True, extract=False):
    if re.match(r'^https?://', remote_url):
        protocol = 'http'
    else:
        protocol = 's3'
        remote_url = prefix_remote_root(remote_url)

    if cache:
        if local_path is None:
            if protocol == 'http':
                filename = urlparse(remote_url).path.split('/')[-1]
            elif protocol == 's3':
                filename = remote_url
            local_path = os.path.join(lore.env.data_dir, filename)
        
        if os.path.exists(local_path):
            return local_path
    elif local_path:
        raise ValueError("You can't pass lore.io.download(local_path=X), unless you also pass cache=True")
    elif extract:
        raise ValueError("You can't pass lore.io.download(extract=True), unless you also pass cache=True")

    with timer('DOWNLOAD: %s' % remote_url):
        temp_file, temp_path = tempfile.mkstemp()
        try:
            if protocol == 'http':
                urlretrieve(remote_url, temp_path)
            else:
                bucket.download_file(remote_url, temp_path)
        except ClientError as e:
            logger.error("Error downloading file: %s" % e)
            raise
            
    if cache:
        dir = os.path.dirname(local_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    
        os.rename(temp_path, local_path)

        if extract:
            with timer('EXTRACT: %s' % local_path, logging.DEBUG):
                with tarfile.open(local_path, 'r:gz') as tar:
                    tar.extractall(os.path.dirname(local_path))
    else:
        local_path = temp_path
        
    return local_path


def upload(local_path, remote_path=None):
    if remote_path is None:
        remote_path = remote_from_local(local_path)
    remote_path = prefix_remote_root(remote_path)
    
    with timer('UPLOAD: %s -> %s' % (local_path, remote_path)):
        try:
            bucket.upload_file(local_path, remote_path, ExtraArgs={'ServerSideEncryption': 'AES256'})
        except ClientError as e:
            logger.error("Error uploading file: %s" % e)
            raise


def remote_from_local(local_path):
    return re.sub(
        r'^%s' % re.escape(lore.env.work_dir),
        '',
        local_path
    )


def prefix_remote_root(path):
    if path.startswith('/'):
        path = path[1:]

    if not path.startswith(lore.env.name + '/'):
        path = os.path.join(lore.env.name, path)

    return path
