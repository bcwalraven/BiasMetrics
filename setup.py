from distutils.core import setup
setup(
  name = 'biasMetrics',      
  packages = ['biasMetrics'],
  version = '0.2',      
  license='MIT',        
  description = 'AUC ROC based classification metrics to determine bias in underrepresented subpopulations.',  
  author = 'Brandon Walraven',
  author_email = '',      
  url = 'https://github.com/bcwalraven',
  download_url = 'https://github.com/bcwalraven/BiasMetrics/archive/v0.02.tar.gz',    
  keywords = ['AUC', 'Classification', 'Metrics'],   
  install_requires=[           
          'numpy',
          'pandas',
          'sklearn'
      ],
)