# NOTICE: ABANDONED

This project has been abandoned. This was initially created as a tool for a game I am working on, but with impending deadlines it simply isn't feasible for me to be spending valuable weeks chasing bugs around when other solutions exist. Anyone is more than welcome to pick up this project -- there are still tons of things that need doing and some reworking that would make this a better native C#/.NET tool.

There are also several other libraries which CREPE depends on, such as hmmlearn and resampy which don't have exact C# equivalents. I am concerned that trying to use replacement libraries may cause minor changes in CREPE's behavior, and re-implementing existing library functions may cause more errors. If I have time in the future I may return to this project.

## About

CrepeSharp is a native C#/.NET implementation of the [CREPE pitch tracker](https://github.com/marl/crepe). It attempts to be as direct of a translation of the original project as possible.

CrepeSharp uses [TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET) and [NumpyDotNet](https://github.com/Quansight-Labs/numpy.net).
