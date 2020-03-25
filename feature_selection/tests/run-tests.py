import os
import unittest
import coverage

COV = coverage.coverage(
  branch=True,
  include='./*',
  omit=[
    './tests/*',
    './*/__init__.py'
  ]
)
COV.start()

tests = unittest.TestLoader().discover('.', pattern='test*.py')
result = unittest.TextTestRunner(verbosity=2).run(tests)

if result.wasSuccessful():
  COV.stop()
  COV.save()
  print('Coverage Summary:')
  COV.report()
  basedir = os.path.abspath(os.path.dirname(__file__))
  covdir = os.path.join(basedir, 'tmp/coverage')
  COV.html_report(directory=covdir)
  print('HTML version: file://%s/index.html' % covdir)
  COV.erase()