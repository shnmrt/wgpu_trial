fn main() {
    pollster::block_on(print_gpu_info())
}

async fn print_gpu_info() {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("no sutiable adapter");

    let info = adapter.get_info();

    println!("{:?}", info);
    println!("Name:        {}", info.name);
    println!("Vendor ID:   {:#06x}", info.vendor);
    println!("Device ID:   {:#06x}", info.device);
    println!("Type:        {:?}", info.device_type); // DiscreteGpu, IntegratedGpu, Cpu, ...
    println!("Backend:     {:?}", info.backend); // Vulkan, Metal, Dx12, Gl, ...
    println!("Driver:      {}", info.driver);
    println!("Driver info: {}", info.driver_info);

    println!("\nFeatures:\n{:#?}", adapter.features());
    println!("\nLimits:\n{:#?}", adapter.limits());
}
