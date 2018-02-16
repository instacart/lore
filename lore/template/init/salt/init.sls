upstart:
  cmd.run:
    - cwd: {{ salt['environ.get']('PWD') }}
    - name: cp upstart/*.conf /etc/init/ && initctl reload-configuration
