use std::borrow::Cow;

use wgpu::util::DeviceExt;

fn main() {
    // pollster::block_on(print_gpu_info())
    pollster::block_on(run())
}

async fn run() {
    // 1. create 1_000_000 elements for vector summation
    const ELEMENT_COUNT: usize = 15_000_000;
    // let mut a_data = vec![1.0f32; ELEMENT_COUNT];
    // let mut b_data = vec![2.0f32; ELEMENT_COUNT];

    let mut a_data = (0..ELEMENT_COUNT)
        .into_iter()
        .map(|v| v as f32)
        .collect::<Vec<f32>>();

    let mut b_data = (0..ELEMENT_COUNT)
        .into_iter()
        .rev()
        .map(|v| v as f32)
        .collect::<Vec<f32>>();

    // pad data to a multiple of workgroup size (256) if necessary
    // 1_000_000 / 256 = 3906.25 so we need 3907 workgroups
    let workgroup_size = 256;
    let num_workgroups = (ELEMENT_COUNT + workgroup_size - 1) / workgroup_size;
    let padded_count = num_workgroups * workgroup_size;

    println!("padded count: {:?}", padded_count);

    a_data.resize(padded_count, 0.0);
    b_data.resize(padded_count, 0.0);

    // 2. initialize webgpu/wgpu instance, adapter, and device
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find an adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    // 3. Load the WGSL compute shader module
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Vector Sum Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // 4. Create and upload buffers
    let buffer_size = (padded_count * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let src_a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer A"),
        contents: bytemuck::cast_slice(&a_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let src_b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer B"),
        contents: bytemuck::cast_slice(&b_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 5. Build Bind Group Layout and Bind Group
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src_b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dst_buffer.as_entire_binding(),
            },
        ],
    });

    // 6. Setup the Compute Pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        ..Default::default()
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // 7. Command Encoding and Dispatching
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // 1_000_000 items / 256 workgroup_size = 3907 workgroups on X dimension.
        // This is safely well under the 65535 Apple M chip structural limit!
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    // Copy results from GPU storage back to mapped staging buffer
    encoder.copy_buffer_to_buffer(&dst_buffer, 0, &staging_buffer, 0, buffer_size);

    // Submit instructions to GPU command queue
    queue.submit(Some(encoder.finish()));

    // 8. Map buffer and read final results back to CPU
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    if let Ok(Ok(())) = receiver.recv() {
        let data = buffer_slice.get_mapped_range().unwrap();
        let result: &[f32] = bytemuck::cast_slice(&data);

        println!("Summation success! First 5 items: {:?}", &result[0..5]);
        println!("Element 999_999: {:?}", result[999_999]);

        drop(data);
        staging_buffer.unmap();
    } else {
        panic!("Failed to map staging buffer!")
    }
}

/// Prints the information on the available GPU on device
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
