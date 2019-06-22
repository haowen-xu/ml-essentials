import unittest
import uuid
from functools import partial

import httpretty
import pytest
import requests
from bson import ObjectId

from mltk import MLStorageClient
from mltk.utils import json_dumps, json_loads


class MLStorageClientTestCase(unittest.TestCase):

    def setUp(self):
        self.client = MLStorageClient('http://127.0.0.1')

    @httpretty.activate
    def test_interface(self):
        c = MLStorageClient('http://127.0.0.1')
        self.assertEqual(c.uri, 'http://127.0.0.1')

        c = MLStorageClient('http://127.0.0.1/')
        self.assertEqual(c.uri, 'http://127.0.0.1')

        # test invalid response should trigger error
        httpretty.register_uri(
            httpretty.POST, 'http://127.0.0.1/v1/_query', body='hello')
        with pytest.raises(IOError,
                           match=r'The response from http://127.0.0.1/v1/'
                                 r'_query\?skip=0 is not JSON: HTTP code is '
                                 r'200'):
            _ = self.client.query()

    @httpretty.activate
    def test_query(self):
        def callback(request, uri, response_headers,
                     expected_body, expected_skip, expected_limit,
                     expected_sort, response_body):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            self.assertEqual(request.querystring['skip'][0], str(expected_skip))
            if expected_limit is not None:
                self.assertEqual(
                    request.querystring['limit'][0], str(expected_limit))
            if expected_sort is not None:
                self.assertEqual(
                    request.querystring['sort'][0], str(expected_sort))
            self.assertEqual(request.body, expected_body)
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, response_body]

        object_ids = [str(ObjectId()) for _ in range(5)]
        docs = [
            {'_id': object_ids[i], 'storage_dir': f'/{object_ids[i]}',
             'uuid': uuid.uuid4()}
            for i in range(len(object_ids))
        ]

        # test bare query
        httpretty.register_uri(
            httpretty.POST,
            'http://127.0.0.1/v1/_query',
            body=partial(
                callback, expected_body=b'{}', expected_skip=0,
                expected_limit=None, expected_sort=None,
                response_body=json_dumps(docs[:2])
            )
        )
        ret = self.client.query()
        self.assertListEqual(ret, docs[:2])

        for obj_id in object_ids[:2]:  # test the storage dir cache
            self.assertEqual(self.client.get_storage_dir(obj_id), f'/{obj_id}')

        with pytest.raises(requests.exceptions.ConnectionError):
            _ = self.client.get_storage_dir(object_ids[2])

        # test query
        httpretty.register_uri(
            httpretty.POST,
            'http://127.0.0.1/v1/_query',
            body=partial(
                callback, expected_body=b'{"name":"hint"}', expected_skip=1,
                expected_limit=99, expected_sort='-start_time',
                response_body=json_dumps(docs[2:4])
            )
        )
        ret = self.client.query(filter={'name': 'hint'}, sort='-start_time',
                                skip=1, limit=99)
        self.assertListEqual(ret, docs[2:4])

    @httpretty.activate
    def test_get(self):
        object_id = str(ObjectId())
        doc = {
            '_id': object_id,
            'storage_dir': f'/{object_id}',
            'uuid': uuid.uuid4(),
        }

        def callback(request, uri, response_headers):
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(doc)]

        httpretty.register_uri(
            httpretty.GET, f'http://127.0.0.1/v1/_get/{object_id}',
            body=callback
        )
        ret = self.client.get(object_id)

        self.assertDictEqual(ret, doc)
        self.assertEqual(self.client.get_storage_dir(object_id),
                         doc['storage_dir'])

    @httpretty.activate
    def test_heartbeat(self):
        def callback(request, uri, response_headers):
            self.assertEqual(request.body, b'')
            response_headers['content-type'] = 'application/json; charset=utf-8'
            heartbeat_received[0] = True
            return [200, response_headers, b'{}']

        heartbeat_received = [False]
        object_id = str(ObjectId())
        httpretty.register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_heartbeat/{object_id}',
            body=callback
        )
        self.client.heartbeat(object_id)
        self.assertTrue(heartbeat_received[0])

    @httpretty.activate
    def test_create_update_delete(self):
        object_id = str(ObjectId())
        doc_fields = {
            'uuid': uuid.uuid4(),
            'name': 'hello',
            'storage_dir': f'/{object_id}',
        }

        # test create
        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(fields, doc_fields)
            o = {'_id': object_id}
            o.update(doc_fields)
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(o)]

        httpretty.register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_create',
            body=callback
        )
        ret = self.client.create(doc_fields)

        expected = {'_id': object_id}
        expected.update(doc_fields)
        self.assertDictEqual(ret, expected)
        self.assertEqual(self.client.get_storage_dir(object_id),
                         doc_fields['storage_dir'])

        # test update
        doc_fields['storage_dir'] = f'/new/{object_id}'

        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(
                fields, {'storage_dir': doc_fields['storage_dir']})
            o = {'_id': object_id}
            o.update(doc_fields)
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(o)]

        httpretty.register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_update/{object_id}',
            body=callback
        )
        ret = self.client.update(
            object_id, {'storage_dir': doc_fields['storage_dir']})

        expected = {'_id': object_id}
        expected.update(doc_fields)
        self.assertDictEqual(ret, expected)
        self.assertEqual(self.client.get_storage_dir(object_id),
                         doc_fields['storage_dir'])

        # test delete
        def callback(request, uri, response_headers):
            self.assertEqual(request.body, b'')
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps([object_id])]

        httpretty.register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_delete/{object_id}',
            body=callback
        )
        self.assertListEqual(self.client.delete(object_id), [object_id])

        with pytest.raises(requests.exceptions.ConnectionError):
            _ = self.client.get_storage_dir(object_id)

    @httpretty.activate
    def test_set_finished(self):
        object_id = str(ObjectId())
        doc_fields = {
            '_id': object_id,
            'uuid': uuid.uuid4(),
            'name': 'hello',
            'status': 'COMPLETED',
            'storage_dir': f'/{object_id}',
        }

        # test set status only
        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(fields, {'status': 'COMPLETED'})
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(doc_fields)]

        httpretty.register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_set_finished/{object_id}',
            body=callback
        )
        ret = self.client.set_finished(object_id, 'COMPLETED')
        self.assertDictEqual(ret, doc_fields)

        # test set status with new fields
        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(
                fields, {'name': 'hello', 'status': 'COMPLETED'})
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(doc_fields)]

        httpretty.register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_set_finished/{object_id}',
            body=callback
        )
        ret = self.client.set_finished(
            object_id, 'COMPLETED', {'name': 'hello'})
        self.assertDictEqual(ret, doc_fields)
        self.assertEqual(
            self.client.get_storage_dir(object_id), f'/{object_id}')

    @httpretty.activate
    def test_get_storage_dir(self):
        object_id = str(ObjectId())
        doc_fields = {'_id': object_id, 'storage_dir': f'/{object_id}'}

        def callback(request, uri, response_headers):
            response_headers['content-type'] = 'application/json; charset=utf-8'
            counter[0] += 1
            return [200, response_headers, json_dumps(doc_fields)]

        counter = [0]
        httpretty.register_uri(
            httpretty.GET, f'http://127.0.0.1/v1/_get/{object_id}',
            body=callback
        )

        self.assertEqual(
            self.client.get_storage_dir(object_id), f'/{object_id}')
        self.assertEqual(counter[0], 1)
        self.assertEqual(
            self.client.get_storage_dir(object_id), f'/{object_id}')
        self.assertEqual(counter[0], 1)

    @httpretty.activate
    def test_get_file(self):
        object_id = str(ObjectId())

        httpretty.register_uri(
            httpretty.GET,
            f'http://127.0.0.1/v1/_getfile/{object_id}/hello.txt',
            body=b'hello, world'
        )
        self.assertEqual(self.client.get_file(object_id, '/./hello.txt'),
                         b'hello, world')
