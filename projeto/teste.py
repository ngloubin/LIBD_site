import unittest
from app import app

class TestApp(unittest.TestCase):
    def test_home_page(self):
        tester = app.test_client()
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Bem-vindo ao Seu Projeto', response.data)

if __name__ == '__main__':
    unittest.main()