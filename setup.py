from distutils.core import setup
setup(
  name = 'mrange',         # How you named your package folder (MyLib)
  packages = ['mrange'],   # Chose the same as "name"
  version = '0.4',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Reduce nested for i in range(n) loops to a single line',   # Give a short description about your library
  author = 'Michael Schilling',                   # Type in your name
  author_email = 'michael@ntropic.de',      # Type in your E-Mail
  url = 'https://github.com/Ntropic',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/Ntropic/mrange/archive/refs/tags/v0.3.tar.gz',    # I explain this later on
  keywords = ['nested', 'loops'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numba',
      ],
)