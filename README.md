# Helen district heat generation modelling

## Setup

### Install Python 3.9 and others (instructions for Ubuntu)

1. Update package list, install prerequisites
    ```sh
    $ sudo apt-get update
    $ sudo apt install software-properties-common
    ```
2. Add "deadsnakes" Python packaging archive (version 3.9 not found in Ubuntu archives) 
    ```sh
    $ sudo add-apt-repository ppa:deadsnakes/ppa
    ```
   Press _enter_ when prompted
3. Install and verify
    ```sh
    $ sudo apt install python3.9 python3.9-distutils
    $ python3.9 --version
    ```
4. Install python distutils, pip and pipenv
    ```sh
    $ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $ python3.9 get-pip.py
    $ python3.9 -m pip install pipenv
    $ echo export PATH="/home/mikko/.local/bin:$PATH" >> ~/.bashrc
    $ export PATH="/home/mikko/.local/bin:$PATH" >> ~/.bashrc
    ```

### Get sources and install python packages

```sh
$ git clone git@github.com:mkouhia/helen-dh-generation-modelling.git
$ cd helen-dh-generation-modelling/
$ pipenv install --dev --skip-lock
```

### Configure dvc

Add s3 credentials, replacing _MY_ACCESS_KEY_ID_ and _MY_SECRET_ACCESS_KEY_ with actual values,
then pull in data
```sh
pipenv shell
dvc remote modify s3-general access_key_id MY_ACCESS_KEY_ID
dvc remote modify s3-general secret_access_key MY_SECRET_ACCESS_KEY
dvc pull
```

### Start jupyter notebook
If you are running Jupyter notebook on a server, to access it on local browser, you must either pipe
the connection via ssh:

```shell
$ ssh -L 8888:localhost:8888 myuser@your_server_ip
```

When logged in to the server, do
```sh
$ pipenv run jupyter notebook
```
Now, on local browser you can access the notebook at URL given by jupyter.


Alternative: start jupyter on server, providing the public IP. This will allow access from
non-localhost.
```sh
$ pipenv run jupyter notebook --ip=your_server_ip
```

