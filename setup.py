from distutils.core import setup
setup(
  name = 'biasMetrics',      
  packages = ['biasMetrics'],
  version = '0.2.5',      
  license='MIT',        
  description = 'AUC ROC based classification metrics to determine bias in underrepresented subpopulations.',  
  author = 'Brandon Walraven',
  author_email = '',      
  url = 'https://github.com/bcwalraven',
  download_url = 'https://github.com/bcwalraven/biasMetrics/archive/v0.2.5.tar.gz',    
  keywords = ['AUC', 'Classification', 'Metrics'],   
  install_requires=[           
          'numpy',
          'pandas',
          'sklearn'
      ],
)