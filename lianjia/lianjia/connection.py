# Copyright (c) 2014, 2019, Oracle and/or its affiliates. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License, version 2.0, as
# published by the Free Software Foundation.
#
# This program is also distributed with certain software (including
# but not limited to OpenSSL) that is licensed under separate terms,
# as designated in a particular file or component or in included license
# documentation.  The authors of MySQL hereby grant you an
# additional permission to link the program and your derivative works
# with the separately licensed software that they have included with
# MySQL.
#
# Without limiting anything contained in the foregoing, this file,
# which is part of MySQL Connector/Python, is also subject to the
# Universal FOSS Exception, version 1.0, a copy of which can be found at
# http://oss.oracle.com/licenses/universal-foss-exception.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License, version 2.0, for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA

"""Connection class using the C Extension
"""

# Detection of abstract methods in pylint is not working correctly
#pylint: disable=W0223

import socket

from . import errors, version
from .catch23 import INT_TYPES
from .constants import (
    CharacterSet, FieldFlag, ServerFlag, ShutdownType, ClientFlag
)
from .abstracts import MySQLConnectionAbstract, MySQLCursorAbstract
from .protocol import MySQLProtocol

HAVE_CMYSQL = False
# pylint: disable=F0401,C0413
try:
    import _mysql_connector
    from .cursor_cext import (
        CMySQLCursor, CMySQLCursorRaw,
        CMySQLCursorBuffered, CMySQLCursorBufferedRaw, CMySQLCursorPrepared,
        CMySQLCursorDict, CMySQLCursorBufferedDict, CMySQLCursorNamedTuple,
        CMySQLCursorBufferedNamedTuple)
    from _mysql_connector import MySQLInterfaceError  # pylint: disable=F0401
except ImportError as exc:
    raise ImportError(
        "MySQL Connector/Python C Extension not available ({0})".format(
            str(exc)
        ))
else:
    HAVE_CMYSQL = True
# pylint: enable=F0401,C0413


class CMySQLConnection(MySQLConnectionAbstract):



    def __init__(self, **kwargs):

        if not HAVE_CMYSQL:
            raise RuntimeError(
                "MySQL Connector/Python C Extension not available")
        self._cmysql = None
        self._columns = []
        self.converter = None
        super(CMySQLConnection, self).__init__(**kwargs)

        if kwargs:
            self.connect(**kwargs)

    def _add_default_conn_attrs(self):

        license_chunks = version.LICENSE.split(" ")
        if license_chunks[0] == "GPLv2":
            client_license = "GPL-2.0"
        else:
            client_license = "Commercial"

        self._conn_attrs.update({
            "_connector_name": "mysql-connector-python",
            "_connector_license": client_license,
            "_connector_version": ".".join(
                [str(x) for x in version.VERSION[0:3]]),
            "_source_host": socket.gethostname()
            })

    def _do_handshake(self):

        self._handshake = {
            'protocol': self._cmysql.get_proto_info(),
            'server_version_original': self._cmysql.get_server_info(),
            'server_threadid': self._cmysql.thread_id(),
            'charset': None,
            'server_status': None,
            'auth_plugin': None,
            'auth_data': None,
            'capabilities': self._cmysql.st_server_capabilities(),
        }

        self._server_version = self._check_server_version(
            self._handshake['server_version_original']
        )

    @property
    def _server_status(self):

        return self._cmysql.st_server_status()

    def set_unicode(self, value=True):

        self._use_unicode = value
        if self._cmysql:
            self._cmysql.use_unicode(value)
        if self.converter:
            self.converter.set_unicode(value)

    @property
    def autocommit(self):

        value = self.info_query("SELECT @@session.autocommit")[0]
        return True if value == 1 else False

    @autocommit.setter
    def autocommit(self, value):  # pylint: disable=W0221

        try:
            self._cmysql.autocommit(value)
            self._autocommit = value
        except MySQLInterfaceError as exc:
            raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                             sqlstate=exc.sqlstate)

    @property
    def database(self):

        return self.info_query("SELECT DATABASE()")[0]

    @database.setter
    def database(self, value):  # pylint: disable=W0221

        self._cmysql.select_db(value)

    @property
    def in_transaction(self):
        return self._server_status & ServerFlag.STATUS_IN_TRANS

    def _open_connection(self):
        charset_name = CharacterSet.get_info(self._charset_id)[0]
        self._cmysql = _mysql_connector.MySQL(  # pylint: disable=E1101,I1101
            buffered=self._buffered,
            raw=self._raw,
            charset_name=charset_name,
            connection_timeout=(self._connection_timeout or 0),
            use_unicode=self._use_unicode,
            auth_plugin=self._auth_plugin)

        if not self.isset_client_flag(ClientFlag.CONNECT_ARGS):
            self._conn_attrs = {}
        cnx_kwargs = {
            'host': self._host,
            'user': self._user,
            'password': self._password,
            'database': self._database,
            'port': self._port,
            'client_flags': self._client_flags,
            'unix_socket': self._unix_socket,
            'compress': self.isset_client_flag(ClientFlag.COMPRESS),
            'ssl_disabled': True,
            "conn_attrs": self._conn_attrs
        }

        tls_versions = self._ssl.get('tls_versions')
        if tls_versions is not None:
            tls_versions.sort(reverse=True)
            tls_versions = ",".join(tls_versions)
        if self._ssl.get('tls_ciphersuites') is not None:
            ssl_ciphersuites = self._ssl.get('tls_ciphersuites')[0]
            tls_ciphersuites = self._ssl.get('tls_ciphersuites')[1]
        else:
            ssl_ciphersuites = None
            tls_ciphersuites = None
        if tls_versions is not None and "TLSv1.3" in tls_versions and \
           not tls_ciphersuites:
            tls_ciphersuites = "TLS_AES_256_GCM_SHA384"
        if not self._ssl_disabled:
            cnx_kwargs.update({
                'ssl_ca': self._ssl.get('ca'),
                'ssl_cert': self._ssl.get('cert'),
                'ssl_key': self._ssl.get('key'),
                'ssl_cipher_suites': ssl_ciphersuites,
                'tls_versions': tls_versions,
                'tls_cipher_suites': tls_ciphersuites,
                'ssl_verify_cert': self._ssl.get('verify_cert') or False,
                'ssl_verify_identity':
                    self._ssl.get('verify_identity') or False,
                'ssl_disabled': self._ssl_disabled
            })

        try:
            self._cmysql.connect(**cnx_kwargs)
        except MySQLInterfaceError as exc:
            raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                             sqlstate=exc.sqlstate)

        self._do_handshake()

    def close(self):

        if self._cmysql:
            try:
                self.free_result()
                self._cmysql.close()
            except MySQLInterfaceError as exc:
                raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                                 sqlstate=exc.sqlstate)
            self._cmysql = None
    disconnect = close

    def is_connected(self):

        if self._cmysql:
            return self._cmysql.ping()

        return False

    def ping(self, reconnect=False, attempts=1, delay=0):
        errmsg = "Connection to MySQL is not available"

        try:
            connected = self._cmysql.ping()
        except AttributeError:
            pass  # Raise or reconnect later
        else:
            if connected:
                return

        if reconnect:
            self.reconnect(attempts=attempts, delay=delay)
        else:
            raise errors.InterfaceError(errmsg)

    def set_character_set_name(self, charset):
        self._cmysql.set_character_set(charset)

    def info_query(self, query):
        self._cmysql.query(query)
        first_row = ()
        if self._cmysql.have_result_set:
            first_row = self._cmysql.fetch_row()
            if self._cmysql.fetch_row():
                self._cmysql.free_result()
                raise errors.InterfaceError(
                    "Query should not return more than 1 row")
        self._cmysql.free_result()

        return first_row

    @property
    def connection_id(self):
        try:
            return self._cmysql.thread_id()
        except MySQLInterfaceError:
            pass  # Just return None

        return None

    def get_rows(self, count=None, binary=False, columns=None, raw=None,
                 prep_stmt=None):
        unread_result = prep_stmt.have_result_set if prep_stmt \
            else self.unread_result
        if not (self._cmysql and unread_result):
            raise errors.InternalError("No result set available")

        if raw is None:
            raw = self._raw

        rows = []
        if count is not None and count <= 0:
            raise AttributeError("count should be 1 or higher, or None")

        counter = 0
        try:
            row = prep_stmt.fetch_row() if prep_stmt \
                else self._cmysql.fetch_row()
            while row:
                if not self._raw and self.converter:
                    row = list(row)
                    for i, _ in enumerate(row):
                        if not raw:
                            row[i] = self.converter.to_python(self._columns[i],
                                                              row[i])
                    row = tuple(row)
                rows.append(row)
                counter += 1
                if count and counter == count:
                    break
                row = prep_stmt.fetch_row() if prep_stmt \
                        else self._cmysql.fetch_row()
            if not row:
                _eof = self.fetch_eof_columns(prep_stmt)['eof']
                if prep_stmt:
                    prep_stmt.free_result()
                    self._unread_result = False
                else:
                    self.free_result()
            else:
                _eof = None
        except MySQLInterfaceError as exc:
            if prep_stmt:
                prep_stmt.free_result()
                raise errors.InterfaceError(str(exc))
            else:
                self.free_result()
                raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                                 sqlstate=exc.sqlstate)

        return rows, _eof

    def get_row(self, binary=False, columns=None, raw=None, prep_stmt=None):
        """Get the next rows returned by the MySQL server"""
        try:
            rows, eof = self.get_rows(count=1, binary=binary, columns=columns,
                                      raw=raw, prep_stmt=prep_stmt)
            if rows:
                return (rows[0], eof)
            return (None, eof)
        except IndexError:
            # No row available
            return (None, None)

    def next_result(self):
        """Reads the next result"""
        if self._cmysql:
            self._cmysql.consume_result()
            return self._cmysql.next_result()
        return None

    def free_result(self):
        """Frees the result"""
        if self._cmysql:
            self._cmysql.free_result()

    def commit(self):
        if self._cmysql:
            self._cmysql.commit()

    def rollback(self):
        if self._cmysql:
            self._cmysql.consume_result()
            self._cmysql.rollback()

    def cmd_init_db(self, database):
        try:
            self._cmysql.select_db(database)
        except MySQLInterfaceError as exc:
            raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                             sqlstate=exc.sqlstate)

    def fetch_eof_columns(self, prep_stmt=None):
        have_result_set = prep_stmt.have_result_set if prep_stmt \
            else self._cmysql.have_result_set
        if not have_result_set:
            raise errors.InterfaceError("No result set")

        fields = prep_stmt.fetch_fields() if prep_stmt \
            else self._cmysql.fetch_fields()
        self._columns = []
        for col in fields:
            self._columns.append((
                col[4],
                int(col[8]),
                None,
                None,
                None,
                None,
                ~int(col[9]) & FieldFlag.NOT_NULL,
                int(col[9])
            ))

        return {
            'eof': {
                'status_flag': self._server_status,
                'warning_count': self._cmysql.st_warning_count(),
            },
            'columns': self._columns,
        }

    def fetch_eof_status(self):
        if self._cmysql:
            return {
                'warning_count': self._cmysql.st_warning_count(),
                'field_count': self._cmysql.st_field_count(),
                'insert_id': self._cmysql.insert_id(),
                'affected_rows': self._cmysql.affected_rows(),
                'server_status': self._server_status,
            }

        return None

    def cmd_stmt_prepare(self, statement):
        if not self._cmysql:
            raise errors.OperationalError("MySQL Connection not available")

        try:
            return self._cmysql.stmt_prepare(statement)
        except MySQLInterfaceError as err:
            raise errors.InterfaceError(str(err))

    # pylint: disable=W0221
    def cmd_stmt_execute(self, prep_stmt, *args):
        try:
            prep_stmt.stmt_execute(*args)
        except MySQLInterfaceError as err:
            raise errors.InterfaceError(str(err))

        self._columns = []
        if not prep_stmt.have_result_set:
            # No result
            self._unread_result = False
            return self.fetch_eof_status()

        self._unread_result = True
        return self.fetch_eof_columns(prep_stmt)

    def cmd_stmt_close(self, prep_stmt):
     if self._unread_result:
        raise errors.InternalError("Unread result found")
        prep_stmt.stmt_close()

    def cmd_stmt_reset(self, prep_stmt):
        if self._unread_result:
            raise errors.InternalError("Unread result found")
        prep_stmt.stmt_reset()
    # pylint: enable=W0221

    def cmd_query(self, query, raw=None, buffered=False, raw_as_string=False):
        """Send a query to the MySQL server"""
        self.handle_unread_result()
        if raw is None:
            raw = self._raw
        try:
            if not isinstance(query, bytes):
                query = query.encode('utf-8')
            self._cmysql.query(query,
                               raw=raw, buffered=buffered,
                               raw_as_string=raw_as_string)
        except MySQLInterfaceError as exc:
            raise errors.get_mysql_exception(exc.errno, msg=exc.msg,
                                             sqlstate=exc.sqlstate)
        except AttributeError:
            if self._unix_socket:
                addr = self._unix_socket
            else:
                addr = self._host + ':' + str(self._port)
            raise errors.OperationalError(
                errno=2055, values=(addr, 'Connection not available.'))

        self._columns = []
        if not self._cmysql.have_result_set:
            # No result
            return self.fetch_eof_status()

        return self.fetch_eof_columns()
    _execute_query = cmd_query

    def cursor(self, buffered=None, raw=None, prepared=None, cursor_class=None,
               dictionary=None, named_tuple=None):
        self.handle_unread_result(prepared)
        if not self.is_connected():
            raise errors.OperationalError("MySQL Connection not available.")
        if cursor_class is not None:
            if not issubclass(cursor_class, MySQLCursorAbstract):
                raise errors.ProgrammingError(
                    "Cursor class needs be to subclass"
                    " of cursor_cext.CMySQLCursor")
            return (cursor_class)(self)

        buffered = buffered or self._buffered
        raw = raw or self._raw

        cursor_type = 0
        if buffered is True:
            cursor_type |= 1
        if raw is True:
            cursor_type |= 2
        if dictionary is True:
            cursor_type |= 4
        if named_tuple is True:
            cursor_type |= 8
        if prepared is True:
            cursor_type |= 16

        types = {
            0: CMySQLCursor,  # 0
            1: CMySQLCursorBuffered,
            2: CMySQLCursorRaw,
            3: CMySQLCursorBufferedRaw,
            4: CMySQLCursorDict,
            5: CMySQLCursorBufferedDict,
            8: CMySQLCursorNamedTuple,
            9: CMySQLCursorBufferedNamedTuple,
            16: CMySQLCursorPrepared
        }
        try:
            return (types[cursor_type])(self)
        except KeyError:
            args = ('buffered', 'raw', 'dictionary', 'named_tuple', 'prepared')
            raise ValueError('Cursor not available with given criteria: ' +
                             ', '.join([args[i] for i in range(5)
                                        if cursor_type & (1 << i) != 0]))

    @property
    def num_rows(self):
        if not self._cmysql.have_result_set:
            raise errors.InterfaceError("No result set")

        return self._cmysql.num_rows()

    @property
    def warning_count(self):
        if not self._cmysql:
            return 0

        return self._cmysql.warning_count()

    @property
    def result_set_available(self):
        if not self._cmysql:
            return False

        return self._cmysql.have_result_set

    @property
    def unread_result(self):
        return self.result_set_available

    @property
    def more_results(self):
        return self._cmysql.more_results()

    def prepare_for_mysql(self, params):
        if isinstance(params, (list, tuple)):
            result = self._cmysql.convert_to_mysql(*params)
        elif isinstance(params, dict):
            result = {}
            for key, value in params.items():
                result[key] = self._cmysql.convert_to_mysql(value)[0]
        else:
            raise ValueError("Could not process parameters")

        return result

    def consume_results(self):
        self._cmysql.consume_result()

    def cmd_change_user(self, username='', password='', database='',
                        charset=45):
        try:
            self._cmysql.change_user(username, password, database)
        except MySQLInterfaceError as exc:
            raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                             sqlstate=exc.sqlstate)

        self._charset_id = charset
        self._post_connection()

    def cmd_refresh(self, options):
        try:
            self._cmysql.refresh(options)
        except MySQLInterfaceError as exc:
            raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                             sqlstate=exc.sqlstate)

        return self.fetch_eof_status()

    def cmd_quit(self):
        self.close()

    def cmd_shutdown(self, shutdown_type=None):
        if not self._cmysql:
            raise errors.OperationalError("MySQL Connection not available")

        if shutdown_type:
            if not ShutdownType.get_info(shutdown_type):
                raise errors.InterfaceError("Invalid shutdown type")
            level = shutdown_type
        else:
            level = ShutdownType.SHUTDOWN_DEFAULT

        try:
            self._cmysql.shutdown(level)
        except MySQLInterfaceError as exc:
            raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                             sqlstate=exc.sqlstate)
        self.close()

    def cmd_statistics(self):
        self.handle_unread_result()

        try:
            stat = self._cmysql.stat()
            return MySQLProtocol().parse_statistics(stat, with_header=False)
        except (MySQLInterfaceError, errors.InterfaceError) as exc:
            raise errors.get_mysql_exception(msg=exc.msg, errno=exc.errno,
                                             sqlstate=exc.sqlstate)

    def cmd_process_kill(self, mysql_pid):
        """Kill a MySQL process"""
        if not isinstance(mysql_pid, INT_TYPES):
            raise ValueError("MySQL PID must be int")
        self.info_query("KILL {0}".format(mysql_pid))

    def handle_unread_result(self, prepared=False):
        """Check whether there is an unread result"""
        unread_result = self._unread_result if prepared is True \
            else self.unread_result
        if self.can_consume_results:
            self.consume_results()
        elif unread_result:
            raise errors.InternalError("Unread result found")
