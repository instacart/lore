import datetime
import inflection
import subprocess
import logging
import os

from sqlalchemy import Column, Float, Integer, String, DateTime, JSON, func, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy import TypeDecorator, types, desc
from sqlalchemy.inspection import inspect
import lore.io
import json

logger = logging.getLogger(__name__)
Base = declarative_base()
adapter = lore.io.metadata.adapter
engine = lore.io.metadata._engine
Session = scoped_session(sessionmaker(bind=engine))


if adapter == 'sqlite':
    # JSON support is not available in SQLite
    class StringJSON(TypeDecorator):
        @property
        def python_type(self):
            return object

        impl = types.String

        def process_bind_param(self, value, dialect):
            return json.dumps(value)

        def process_literal_param(self, value, dialect):
            return value

        def process_result_value(self, value, dialect):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return None
    JSON = StringJSON

    # Commenting sqlite queries with the SQLAlchemy declarative_base API
    # is broken: https://github.com/sqlalchemy/sqlalchemy/issues/4396
    engine.dialect.supports_sane_rowcount = False
    engine.dialect.supports_sane_multi_rowcount = False  # for executemany()


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
    def get(cls, *key):
        session = Session()

        filter = {str(k.name): v for k, v in dict(zip(inspect(cls).primary_key, key)).items()}
        instance = session.query(cls).filter_by(**filter).first()
        session.close()
        return instance

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

    @classmethod
    def all(cls, order_by=None, limit=None, **filters):
        session = Session()
        query = session.query(cls)
        if filters:
            query = query.filter_by(**filters)
        if isinstance(order_by, list) or isinstance(order_by, tuple):
            query = query.order_by(*order_by)
        elif order_by is not None:
            query = query.order_by(order_by)
        if limit:
            query = query.limit(limit)
        result = query.all()
        session.close()
        return result

    @classmethod
    def last(cls, order_by=None, limit=1, **filters):
       if order_by is None:
           order_by = inspect(cls).primary_key
       if isinstance(order_by, list) or isinstance(order_by, tuple):
           order_by = desc(*order_by)
       else:
           order_by = desc(order_by)
       return cls.first(order_by=order_by, limit=limit, **filters)

    @classmethod
    def first(cls, order_by=None, limit=1, **filters):
        if order_by is None:
            order_by = inspect(cls).primary_key
        result = cls.all(order_by=order_by, limit=limit, **filters)

        if limit == 1:
            if len(result) == 0:
                result = None
            else:
                result = result[0]

        return result

    def save(self):
        session = Session()
        session.add(self)
        try:
            return session.commit()
        except Exception as ex:
            session.rollback()
            raise

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        return self.save()

    def delete(self):
        session = Session()
        session.delete(self)
        try:
            return session.commit()
        except Exception as ex:
            session.rollback()
            raise


class Commit(Crud, Base):
    sha = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.now)
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
            '--format=NAME: %an%nEMAIL: %aE%nDATE: %at%nMESSAGE:%n%B',
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

            return Commit.get_or_create(
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
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.now)
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
    id = Column(Integer, primary_key=True)
    commit_sha = Column(String, ForeignKey('commits.sha'))
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.now)
    completed_at = Column(DateTime)
    snapshot_id = Column(Integer, ForeignKey('snapshots.id'), nullable=False, index=True)
    train = Column(Float)
    validate = Column(Float)
    test = Column(Float)
    score = Column(Float)
    iterations = Column(Integer)
    model = Column(String, index=True)
    args = Column(JSON)
    stats = Column(JSON)
    custom_data = Column(JSON)
    url = Column(String)
    uploaded_at = Column(DateTime)

    commit = relationship('Commit', back_populates='fittings')
    predictions = relationship('Prediction', back_populates='fitting')
    snapshot = relationship('Snapshot', back_populates='fittings')

    def __init__(self, **kwargs):
        if 'commit' not in kwargs:
            self.commit = Commit.from_git()
        super(Fitting, self).__init__(**kwargs)


class Prediction(Crud, Base):
    id = Column(Integer, primary_key=True)
    fitting_id = Column(Integer, ForeignKey('fittings.id'), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.now)
    value = Column(JSON)
    key = Column(JSON)
    features = Column(JSON)
    custom_data = Column(JSON)

    fitting = relationship('Fitting', back_populates='predictions')


Base.metadata.create_all(engine)
