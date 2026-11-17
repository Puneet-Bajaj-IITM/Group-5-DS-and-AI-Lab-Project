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
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
