pipeline {
    agent any
    
    environment {
          VENV_PATH = "myprojectenv"
          
    }

    stages {
        stage('prepare') {
            steps {
                script {
                    sh 'bash -c "python3 -m venv $VENV_PATH" '
                    sh 'bash -c "source  $VENV_PATH/bin/activate" '
                    
                }pipeline {
        agent any 
                    
        environment {
                        VIRTUAL_ENV_DIR = 'myprojectenv'
                        FLASK_APP = 'main.py'
                    }
                }
            }
        }
    }
}
