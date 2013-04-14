Pod::Spec.new do |s|
	s.name		= "TCNeuralNetwork"
	s.version	= "1.0"
	s.summary 	= "A simple neural network library in Objective -C."
	s.homepage	= "https://github.com/theocalmes/TCNeuralNetwork.git"
	s.license	= 'MIT'
	s.author 	= {"Theodore Calmes" => "theo@thoughtbot.com"}
	s.source    = { 
    	:git => "https://github.com/theocalmes/TCNeuralNetwork.git",
    	:tag => "0.0.2"
  	}
  	s.source_files = 'NeuralNetwork/**/*.{m,h}'
	s.requires_arc = true
	s.framework    = 'Accelerate'
	s.platform     = :ios, '5.0'
end
