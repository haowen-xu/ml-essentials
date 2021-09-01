pipeline {
  agent any

  stages {
    stage('deploy quant-tradesys-api') {
      steps {
        sh './ci-scripts/deploy.sh quant-tradesys-api'
      }
    }
    stage('deploy quant-tradesys-testbed1') {
      steps {
        sh './ci-scripts/deploy.sh quant-tradesys-testbed1'
      }
    }
    stage('deploy quant-tradesys-pipeline') {
      steps {
        sh './ci-scripts/deploy.sh quant-tradesys-pipeline'
      }
    }
  }
}
