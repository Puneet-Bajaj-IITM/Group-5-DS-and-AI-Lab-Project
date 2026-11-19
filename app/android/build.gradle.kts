allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
// permission_handler_android asks for compileSdk 37, which the SDK manager
// ships as "android-37.0"; AGP 9.1 looks for "android-37" and fails.
subprojects {
    val pointAtInstalledSdk = { p: Project ->
        p.extensions.findByType(com.android.build.gradle.BaseExtension::class.java)?.let {
            if (it.compileSdkVersion == "android-37") it.compileSdkVersion("android-37.0")
        }
    }
    if (state.executed) pointAtInstalledSdk(this) else afterEvaluate { pointAtInstalledSdk(this) }
}
// camera_android_camerax compiles against camera-core 1.5.x, whose API jar
// references CallbackToFutureAdapter from concurrent-futures; without it on
// the plugin's compile classpath javac fails ("class file ... not found").
subprojects {
    if (name == "camera_android_camerax") {
        pluginManager.withPlugin("com.android.library") {
            dependencies.add("implementation", "androidx.concurrent:concurrent-futures:1.2.0")
        }
    }
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
