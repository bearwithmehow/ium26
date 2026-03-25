pipeline {
    agent any

    parameters{
        string(defaultValue: '1000', name: 'CUTOFF')
        string(defaultValue: 'bearwithmehow', name: 'KAGGLE_USERNAME')
        password(defaultValue: '', name: 'KAGGLE_KEY')
    }

    stages {
        stage('checkout: Check out from version control') {
            steps {
                checkout scm
            }
        }

        stage('sh: Shell Script') {
            steps {
                withEnv([
                    "KAGGLE_USERNAME=${params.KAGGLE_USERNAME}",
                    "KAGGLE_KEY=${params.KAGGLE_KEY}",
                    "CUTOFF=${params.CUTOFF}"
                ]) {
                    sh 'echo KAGGLE_USERNAME: $KAGGLE_USERNAME'
                    sh 'echo CUTOFF: $CUTOFF'
                    sh 'chmod +x script.sh'
                    sh './script.sh'
                }
            }
        }

        stage('archiveArtifacts') {
            steps {
                archiveArtifacts artifacts: 'artifacts/**/*', fingerprint: true
            }
        }
    }
}