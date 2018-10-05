import datetime
import inflection
import subprocess
import logging
import os

from sqlalchemy import Column, Float, Integer, String, DateTime, JSON, func, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, relationship, scoped_session

import lore.io

logger = logging.getLogger(__name__)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=lore.io.metadata._engine))


class Crud(object):
    query = Session.query_property()

    @declared_attr
    def __tablename__(cls):
        return inflection.pluralize(inflection.underscore(cls.__name__))

    def __repr__(self):
        properties = ['%s=%s' % (key, value) for key, value in self.__dict__.items() if key[0] != '_']
        return '<%s(%s)>' % (self.__class__.__name__, ', '.join(properties))

    @classmethod
    def create(cls, **kwargs):
        self = cls(**kwargs)
        self.save()
        return self

    @classmethod
    def get_or_create(cls, **kwargs):
        '''
        Creates an object or returns the object if exists
        credit to Kevin @ StackOverflow
        from: http://stackoverflow.com/questions/2546207/does-sqlalchemy-have-an-equivalent-of-djangos-get-or-create
        '''
        session = Session()
        instance = session.query(cls).filter_by(**kwargs).first()
        session.close()

        if not instance:
            self = cls(**kwargs)
            self.save()
        else:
            self = instance

        return self

    def save(self):
        session = Session()
        session.add(self)
        return session.commit()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        return self.save()

    def delete(self):
        session = Session()
        session.delete(self)
        return session.commit()


class Commit(Crud, Base):
    sha = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now())
    message = Column(String)
    author_name = Column(String, index=True)
    author_email = Column(String)
    fittings = relationship('Fitting', back_populates='commit')
    snapshots = relationship('Snapshot', back_populates='commit')

    @classmethod
    def from_git(cls, sha='HEAD'):
        process = subprocess.Popen([
            'git',
            'rev-list',
            '--format=NAME: %an%nEMAIL: %aE%nDATE: %at%nMESSAGE:%N%B',
            '--max-count=1',
            sha,
        ], stdout=subprocess.PIPE)
        out, err = process.communicate()

        # If there is no Git repo, exit code will be non-zero
        if process.returncode == 0:
            lines = out.strip().decode().split(os.linesep)

            check, sha = lines[0].split('commit ')
            if check or not sha:
                logger.error('bad git parse: %s' % out)

            check, author_name = lines[1].split('NAME: ')
            if check or not author_name:
                logger.error('bad git parse for NAME: %s' % out)

            check, author_email = lines[2].split('EMAIL: ')
            if check or not author_email:
                logger.error('bad git parse for EMAIL: %s' % out)

            check, date = lines[3].split('DATE: ')
            if check or not date:
                logger.error('bad git parse for DATE: %s' % out)
            created_at = datetime.datetime.fromtimestamp(int(date))

            check, message = lines[4], os.linesep.join(lines[5:])
            if check != 'MESSAGE:' or not message:
                logger.error('bad git parse for MESSAGE: %s' % out)

            return Commit(
                sha=sha,
                author_name=author_name,
                author_email=author_email,
                created_at=created_at,
                message=message
            )
        else:
            return None


class Snapshot(Crud, Base):
    """
    Metadata summary description of each column in the snapshot

    """
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    completed_at = Column(DateTime)
    pipeline = Column(String, index=True)
    cache = Column(String)
    args = Column(String)
    commit_sha = Column(String, ForeignKey('commits.sha'), index=True)
    # samples = Column(Integer)
    bytes = Column(Integer)
    head = Column(String)
    tail = Column(String)
    stats = Column(String)
    encoders = Column(JSON)

    description = Column(String)
    fittings = relationship('Fitting', back_populates='snapshot')
    commit = relationship('Commit', back_populates='snapshots')


class Fitting(Crud, Base):
    name = Column(String, primary_key=True)
    commit_sha = Column(String, ForeignKey('commits.sha'))
    created_at = Column(DateTime, nullable=False, default=func.now())
    completed_at = Column(DateTime)
    snapshot_id = Column(Integer, ForeignKey('snapshots.id'), nullable=False, index=True)
    train = Column(Float)
    validate = Column(Float)
    test = Column(Float)
    score = Column(Float)
    iterations = Column(Integer)
    model = Column(String, index=True)
    args = Column(JSON())
    stats = Column(JSON())
    custom_data = Column(JSON())
    url = Column(String)
    uploaded_at = Column(DateTime)

    commit = relationship('Commit', back_populates='fittings')
    predictions = relationship('Prediction', back_populates='fitting')
    snapshot = relationship('Snapshot', back_populates='fittings')

    def __init__(self, **kwargs):
        self.commit = Commit()
        super(Fitting, self).__init__(**kwargs)


class Prediction(Crud, Base):
    id = Column(Integer, primary_key=True)
    fitting_name = Column(String, ForeignKey('fittings.name'), nullable=False, index=True)
    created_at = Column(DateTime, default=func.now())
    value = Column(JSON())
    key = Column(JSON())
    features = Column(JSON())
    custom_data = Column(JSON())

    fitting = relationship('Fitting', back_populates='predictions')


Base.metadata.create_all(lore.io.metadata._engine)
