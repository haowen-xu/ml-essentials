pipeline {
  agent any

  stages {
    stage('deploy') {
      steps {
        sh './ci-scripts/deploy.sh quant-tradesys-api'
        sh './ci-scripts/deploy.sh quant-tradesys-testbed1'
        sh './ci-scripts/deploy.sh quant-tradesys-pipeline'
      }
    }
  }
}
