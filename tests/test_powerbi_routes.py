"""Test reporting routes without starting the app's ML/database initialization."""
import ast
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


class RouteTests(unittest.TestCase):
    def test_setup_exports_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'powerbi/exports').mkdir(parents=True)
            (base / 'powerbi/README.md').write_text('Setup instructions')
            (base / 'powerbi/exports/documents.csv').write_text('document_id,title\n1,Folio\n')
            app = FastAPI()
            namespace = dict(app=app, Path=Path, BASE_DIR=directory, FileResponse=FileResponse, HTTPException=HTTPException)
            module = ast.parse((ROOT / 'backend/server.py').read_text(encoding='utf-8'))
            routes = ast.Module(body=[node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in {'powerbi_setup', 'powerbi_export_file'}], type_ignores=[])
            exec(compile(routes, 'reporting_routes', 'exec'), namespace)
            with TestClient(app) as client:
                self.assertIn('Setup instructions', client.get('/powerbi/setup').text)
                self.assertIn('Folio', client.get('/powerbi/exports/documents.csv').text)
                self.assertEqual(client.get('/powerbi/exports/ai_analytics.csv').status_code, 404)
                self.assertEqual(client.get('/powerbi/exports/passwords.csv').status_code, 404)


if __name__ == '__main__':
    unittest.main()
